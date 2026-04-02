#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import subprocess
from pathlib import Path
import tempfile
import re

JSONL_PATH = Path(os.environ.get("JSONL_PATH", "/path/to/infer_results.jsonl"))
OUT_ROOT   = Path(os.environ.get("OUT_ROOT", "/path/to/render_results"))
SIF_PATH   = os.environ.get("SIF_PATH", "/path/to/tikz.sif")

DENSITY = 300           # Output image DPI for conversion
CROP_MARGIN_PT = 10     # Retain 10pt white margin after cropping for non-standalone

# ---- NEW: pdflatex timeout (seconds) ----
PDFLATEX_TIMEOUT_SEC = 30  

LATEX_BLOCK_FULL_RE = re.compile(r"```latex(.*?)```", re.DOTALL | re.IGNORECASE)
LATEX_BLOCK_START_RE = re.compile(r"```latex\s*", re.IGNORECASE)

DOC_CLASS_RE = re.compile(r"\\documentclass(\[[^\]]*\])?\{([^}]*)\}", re.IGNORECASE)
BEGIN_DOC_RE = re.compile(r"\\begin\{document\}", re.IGNORECASE)
PAGESTYLE_EMPTY_RE = re.compile(r"\\pagestyle\{empty\}|\\thispagestyle\{empty\}", re.IGNORECASE)

def ensure_outdir(jsonl_path: Path, out_root: Path) -> Path:
    """Ensure the output directory and its subfolders exist."""
    out_dir = out_root / jsonl_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["pdf", "png", "tex", "log"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    return out_dir

def format_id_6(sid) -> str:
    """Format the id as a 6-digit zero-padded string, or make alphanumeric safe fallback."""
    if sid is None:
        return "000000"
    if isinstance(sid, int):
        return f"{sid:06d}"
    s = str(sid).strip()
    if s.isdigit():
        return f"{int(s):06d}"
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s[:128] if len(s) > 128 else s

def extract_latex_best_effort(text: str):
    """
    Return: (code, status)
      status:
        - "full_block"    Found full ```latex ... ```
        - "open_block"    Found ```latex but no closing ```, likely truncated
        - "no_block"      No ```latex block
        - "empty"         Output is empty
    """
    if not text:
        return "", "empty"

    m = LATEX_BLOCK_FULL_RE.search(text)
    if m:
        return m.group(1).strip(), "full_block"

    m2 = LATEX_BLOCK_START_RE.search(text)
    if m2:
        code = text[m2.end():].strip()
        return code, "open_block"

    return "", "no_block"

def summarize_tex_problem(status: str, code: str) -> str:
    """Summarize issues detected when extracting LaTeX code."""
    if status == "open_block":
        return "LaTeX code block is not closed (``` missing). Output likely truncated."
    if status == "no_block":
        return "Missing ```latex ...``` block."
    if not code.strip():
        return "Empty LaTeX code."
    return ""

def is_standalone_doc(code: str) -> bool:
    """
    Determine if documentclass is 'standalone'. Only applies to complete documents.
    """
    if not code:
        return False
    m = DOC_CLASS_RE.search(code)
    if not m:
        return False
    cls = (m.group(2) or "").strip().lower()
    return "standalone" in cls

def inject_pagestyle_empty(code: str) -> str:
    """
    Inject page style removal for non-standalone documents to avoid page number in cropping bbox.
    - If code already has pagestyle{empty}/thispagestyle{empty}, do nothing.
    - Inject after \begin{document}.
    """
    if not code:
        return code
    if PAGESTYLE_EMPTY_RE.search(code):
        return code
    m = BEGIN_DOC_RE.search(code)
    if not m:
        return code
    insert = "\n\\pagestyle{empty}\\thispagestyle{empty}\n"
    return code[:m.end()] + insert + code[m.end():]

# --------------------------
# NEW: make pdflatex timeout non-fatal
# --------------------------
def run_pdflatex(tex_dir: Path, tex_basename: str) -> subprocess.CompletedProcess:
    """
    Run pdflatex inside apptainer.

    Key behavior change:
      - If pdflatex times out, DO NOT raise; return a CompletedProcess with returncode=124,
        and stderr appended with [TIMEOUT] marker, so the caller's existing failure branch works.
    """
    cmd = [
        "apptainer", "exec", SIF_PATH,
        "pdflatex", "--shell-escape",
        "-interaction=nonstopmode", "-halt-on-error",
        f"{tex_basename}.tex",
    ]
    try:
        return subprocess.run(
            cmd, cwd=tex_dir,
            capture_output=True, encoding="latin-1", text=True,
            timeout=PDFLATEX_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode("latin-1", errors="ignore") if e.stdout else "")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode("latin-1", errors="ignore") if e.stderr else "")
        if stderr:
            stderr = stderr + "\n"
        stderr = stderr + f"[TIMEOUT] pdflatex exceeded {PDFLATEX_TIMEOUT_SEC}s"
        return subprocess.CompletedProcess(cmd, returncode=124, stdout=stdout, stderr=stderr)

def run_pdfcrop(in_pdf: Path, out_pdf: Path, margin_pt: int) -> subprocess.CompletedProcess:
    """
    Run pdfcrop within the container. Margins are in pt units.
    """
    cmd = [
        "apptainer", "exec", SIF_PATH,
        "pdfcrop",
        "--margins", str(margin_pt),
        str(in_pdf),
        str(out_pdf),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)

def pdf_to_png_no_crop(pdf_path: Path, png_path: Path, density: int = DENSITY) -> subprocess.CompletedProcess:
    """Convert PDF page to PNG without any cropping."""
    cmd = [
        "convert",
        "-density", str(density),
        str(pdf_path) + "[0]",
        "-background", "white", "-flatten",
        "-quality", "95",
        str(png_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)

def pdf_to_png_trim_border(pdf_path: Path, png_path: Path, density: int = DENSITY, margin_pt: int = CROP_MARGIN_PT) -> subprocess.CompletedProcess:
    """
    Fallback: If pdfcrop fails, use convert's -trim + border as a backup.
    Note: border can only use pixels; here pt is converted to px: px = density * pt / 72
    """
    border_px = int(round(density * margin_pt / 72.0))  # 10pt at 300dpi -> ~42px
    cmd = [
        "convert",
        "-density", str(density),
        str(pdf_path) + "[0]",
        "-background", "white", "-flatten",
        # fuzz allows for non-pure white background/antialiased edges, prevents trim failure
        "-fuzz", "1%",
        "-trim", "+repage",
        "-bordercolor", "white", "-border", str(border_px),
        "-quality", "95",
        str(png_path),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)

def clean_aux(tex_dir: Path, tex_basename: str):
    """Remove auxiliary files after compilation."""
    for ext in [".aux", ".log", ".out", ".toc"]:
        (tex_dir / f"{tex_basename}{ext}").unlink(missing_ok=True)

def main():
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"JSONL does not exist: {JSONL_PATH}")

    out_dir = ensure_outdir(JSONL_PATH, OUT_ROOT)
    print(f"[INFO] Output directory: {out_dir}")

    lines = JSONL_PATH.read_text(encoding="utf-8").splitlines()
    success_items = []
    fail_items = []

    with tempfile.TemporaryDirectory(prefix="tikz_build_") as tmpd:
        tmp_dir = Path(tmpd)

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                fail_items.append({"id": f"line:{idx}", "error": f"JSON parse error: {e}"})
                continue

            item_id = obj.get("id", idx)
            fid = format_id_6(item_id)

            src_image_path = obj.get("image_path", "")
            question = obj.get("question", "")
            outputs = obj.get("outputs", [])

            output_text = outputs[0] if outputs else ""
            code, status = extract_latex_best_effort(output_text)

            # First, determine whether it is standalone (based on raw code)
            is_standalone = is_standalone_doc(code)

            # Always write raw code to tex (do not inject), regardless of whether it is compilable
            tex_out_path = out_dir / "tex" / f"{fid}.tex"
            if code.strip():
                tex_out_path.write_text(code, encoding="utf-8")
            else:
                tex_out_path.write_text("% [EMPTY OR MISSING LATEX BLOCK]\n", encoding="utf-8")

            # If there is no compilable code, fail immediately (but tex is already saved)
            pre_err = summarize_tex_problem(status, code)
            if pre_err:
                fail_items.append({
                    "id": item_id,
                    "error": pre_err,
                    "src_image_path": src_image_path,
                    "question": question,
                    "output_text": output_text[:4000],
                    "tex_path": str(tex_out_path.resolve()),
                })
                continue

            # For non-standalone: inject page number removal for compilation to avoid cropping issues
            code_for_compile = code
            if not is_standalone:
                code_for_compile = inject_pagestyle_empty(code)

            # Compile in the tmp directory (using the same fid)
            tex_tmp_path = tmp_dir / f"{fid}.tex"
            tex_tmp_path.write_text(code_for_compile, encoding="utf-8")

            result = run_pdflatex(tmp_dir, fid)

            compile_log_path = out_dir / "log" / f"{fid}.compile.log.txt"
            compile_log_path.write_text(
                (result.stdout or "") + "\n==== STDERR ====\n" + (result.stderr or ""),
                encoding="utf-8"
            )

            if result.returncode != 0:
                clean_aux(tmp_dir, fid)

                full_log = (result.stdout or "") + "\n==== STDERR ====\n" + (result.stderr or "")
                log_lines = full_log.splitlines()
                error_lines = []
                for l in log_lines:
                    st = l.strip()
                    if st.startswith("!") or st.startswith("l.") or st.startswith("L.") or st.startswith("[TIMEOUT]"):
                        error_lines.append(l)
                if not error_lines:
                    error_lines = log_lines[-20:]
                error_text = "\n".join(error_lines)
                if len(error_text) > 3000:
                    error_text = error_text[:3000] + "\n...[log truncated]"

                fail_items.append({
                    "id": item_id,
                    "error": error_text,
                    "src_image_path": src_image_path,
                    "question": question,
                    "output_text": output_text[:4000],
                    "tex_path": str(tex_out_path.resolve()),
                    "log_path": str(compile_log_path.resolve()),
                })
                continue

            # PDF copy out
            src_pdf = tmp_dir / f"{fid}.pdf"
            dst_pdf = out_dir / "pdf" / f"{fid}.pdf"
            if src_pdf.exists():
                shutil.copy2(src_pdf, dst_pdf)
            else:
                fail_items.append({
                    "id": item_id,
                    "error": "PDF missing after successful pdflatex return code (unexpected).",
                    "src_image_path": src_image_path,
                    "question": question,
                    "tex_path": str(tex_out_path.resolve()),
                    "log_path": str(compile_log_path.resolve()),
                })
                continue

            # For non-standalone: run pdfcrop (retain 10pt white margin)
            cropped_ok = True
            if not is_standalone:
                cropped_pdf = out_dir / "pdf" / f"{fid}.cropped.pdf"
                crop_res = run_pdfcrop(dst_pdf, cropped_pdf, CROP_MARGIN_PT)

                crop_log_path = out_dir / "log" / f"{fid}.pdfcrop.log.txt"
                crop_log_path.write_text(
                    (crop_res.stdout or "") + "\n==== STDERR ====\n" + (crop_res.stderr or ""),
                    encoding="utf-8"
                )

                if crop_res.returncode == 0 and cropped_pdf.exists():
                    # Replace main pdf with cropped pdf
                    try:
                        dst_pdf.unlink(missing_ok=True)
                        cropped_pdf.rename(dst_pdf)
                    except Exception:
                        # If rename fails (e.g., cross-device), copy and then remove
                        shutil.copy2(cropped_pdf, dst_pdf)
                        cropped_pdf.unlink(missing_ok=True)
                else:
                    cropped_ok = False  # PNG generation will use fallback trim+border

            # PNG output
            dst_png = out_dir / "png" / f"{fid}.png"

            if is_standalone:
                # Standalone: do not crop, convert as is
                conv_res = pdf_to_png_no_crop(dst_pdf, dst_png, density=DENSITY)
            else:
                # Non-standalone: prefer to use pdf already cropped by pdfcrop (no trim)
                # If pdfcrop fails: fallback to convert with trim + border (margin_pt in px)
                if cropped_ok:
                    conv_res = pdf_to_png_no_crop(dst_pdf, dst_png, density=DENSITY)
                else:
                    conv_res = pdf_to_png_trim_border(dst_pdf, dst_png, density=DENSITY, margin_pt=CROP_MARGIN_PT)

            if conv_res.returncode != 0 or not dst_png.exists():
                err = (conv_res.stderr or conv_res.stdout or "").strip()
                fail_items.append({
                    "id": item_id,
                    "error": f"PDF→PNG conversion failed: {err[:2000]}",
                    "src_image_path": src_image_path,
                    "question": question,
                    "tex_path": str(tex_out_path.resolve()),
                    "pdf_path": str(dst_pdf.resolve()),
                    "log_path": str(compile_log_path.resolve()),
                })
            else:
                success_items.append({
                    "id": item_id,
                    "image_path": str(dst_png.resolve()),
                    "src_image_path": src_image_path,
                    "question": question,
                    "tex_path": str(tex_out_path.resolve()),
                    "pdf_path": str(dst_pdf.resolve()),
                    "log_path": str(compile_log_path.resolve()),
                    "is_standalone": bool(is_standalone),
                    "cropped": bool((not is_standalone) and cropped_ok),
                })

            clean_aux(tmp_dir, fid)
            for ext in (".pdf", ".tex"):
                (tmp_dir / f"{fid}{ext}").unlink(missing_ok=True)

    success_jsonl = out_dir / "success.jsonl"
    with success_jsonl.open("w", encoding="utf-8") as f:
        for item in success_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    failed_jsonl = out_dir / "failed.jsonl"
    with failed_jsonl.open("w", encoding="utf-8") as f:
        for item in fail_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] success={len(success_items)} failed={len(fail_items)}")
    print(f"[INFO] success.jsonl: {success_jsonl}")
    print(f"[INFO] failed.jsonl:  {failed_jsonl}")
    print(f"[INFO] outputs saved under: {out_dir}")

if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    dt = time.time() - t0
    m, s = divmod(dt, 60)
    print(f"[TIMER] {int(m)} min {s:.2f} sec")
