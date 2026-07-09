#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import tempfile
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ================= Configuration Paths =================
DISTILL_BASE = Path(os.environ.get("DISTILL_BASE", "/path/to/distill/dir"))
SIF_PATH = os.environ.get("SIF_PATH", "/path/to/tikz.sif")

# Default parameters
BATCH_SIZE = 64
TIMEOUT_COMPILE_SEC = 10
TIMEOUT_CONVERT_SEC = 10
DPI = 300


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, required=True, help="Split index (0-9)")
    p.add_argument("--base", type=str, default=str(DISTILL_BASE), help="Base distill dir (or set DISTILL_BASE env var)")
    p.add_argument("--sif", type=str, default=SIF_PATH, help="Apptainer sif path (or set SIF_PATH env var)")

    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--timeout_compile", type=int, default=TIMEOUT_COMPILE_SEC)
    p.add_argument("--timeout_convert", type=int, default=TIMEOUT_CONVERT_SEC)
    p.add_argument("--dpi", type=int, default=DPI)

    p.add_argument("--shell_escape", action="store_true", help="Enable --shell-escape for pdflatex")
    p.add_argument("--dump_png_dir", type=str, default="", help="Optional: also dump png files to this dir")

    # Option to only process status=ok
    p.add_argument("--only_ok", action="store_true", help="Only compile lines with status=='ok'")

    # Error log real-time control
    p.add_argument("--fsync_every", type=int, default=1, help="fsync error log every N error lines (default: 1)")
    return p.parse_args()


def _run(cmd, cwd: Path, env: dict, timeout: int):
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    out = p.stdout.decode("latin-1", errors="ignore")
    err = p.stderr.decode("latin-1", errors="ignore")
    return p.returncode, out, err


class ErrorWriter:
    """
    Write error log using os.open/os.write to avoid Python buffering causing "non-real-time" appearance
    """
    def __init__(self, path: Path, fsync_every: int = 1):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self.n = 0
        self.fsync_every = max(1, int(fsync_every))

    def write_obj(self, obj: dict):
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8", errors="ignore")
        os.write(self.fd, line)
        self.n += 1
        if self.n % self.fsync_every == 0:
            os.fsync(self.fd)

    def close(self):
        try:
            os.fsync(self.fd)
        except Exception:
            pass
        try:
            os.close(self.fd)
        except Exception:
            pass


def compile_and_render(code: str, workdir: Path, sif: str, shell_escape: bool, dpi: int,
                       timeout_compile: int, timeout_convert: int) -> bytes:
    """
    Compile code as-is (no wrap, no fix, no injection), then render PNG on success, return png bytes
    """
    tex_path = workdir / "main.tex"
    pdf_path = workdir / "main.pdf"
    png_path = workdir / "main.png"

    tex_path.write_text(code, encoding="utf-8", errors="ignore")

    # Environment variable isolation (avoid TEXMF cache conflicts)
    texmf_var = workdir / "texmf_var"
    texmf_var.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TEXMFVAR"] = str(texmf_var)
    env["TEXMFHOME"] = str(texmf_var)

    pdflatex_cmd = [
        "apptainer", "exec", sif,
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
    ]
    if shell_escape:
        pdflatex_cmd.append("--shell-escape")
    pdflatex_cmd.append("main.tex")

    rc, out, err = _run(pdflatex_cmd, cwd=workdir, env=env, timeout=timeout_compile)
    if rc != 0 or (not pdf_path.exists()):
        msg = ""
        lines = out.splitlines()
        bang = [l.strip() for l in lines if l.strip().startswith("!")]
        if bang:
            msg = " | ".join(bang[:3])
        elif "not found" in out.lower():
            cand = [l.strip() for l in lines if ("not found" in l.lower()) or ("File" in l and "not found" in l)]
            msg = " | ".join(cand[:3]) if cand else ""
        if not msg:
            msg = (err.strip() or "Compile failed (unknown)")
        raise RuntimeError(msg)

    convert_cmd = [
        "apptainer", "exec", sif,
        "convert",
        "-density", str(dpi),
        str(pdf_path) + "[0]",
        "-background", "white",
        "-flatten",
        "-quality", "95",
        str(png_path),
    ]
    rc2, out2, err2 = _run(convert_cmd, cwd=workdir, env=env, timeout=timeout_convert)
    if rc2 != 0 or (not png_path.exists()):
        msg = (err2.strip() or "PNG generation failed (convert)")
        raise RuntimeError(msg)

    return png_path.read_bytes()


def main():
    args = parse_args()

    base = Path(args.base)
    idx = args.idx
    in_jsonl = base / f"{idx}.jsonl"
    out_parquet = base / f"{idx}_runtime_validation.parquet"
    err_jsonl = base / f"error_{idx}.jsonl"

    if not in_jsonl.exists():
        raise FileNotFoundError(f"Input jsonl not found: {in_jsonl}")

    dump_dir = Path(args.dump_png_dir) if args.dump_png_dir else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    # Parquet schema: id, image(struct{bytes,path}), code
    img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema([
        ("id", pa.string()),
        ("image", img_type),
        ("code", pa.string()),
    ])
    writer = pq.ParquetWriter(str(out_parquet), schema=schema, compression="zstd")

    ok_cnt = 0
    fail_cnt = 0
    skipped_cnt = 0

    buf_id, buf_code, buf_bytes, buf_path = [], [], [], []

    def flush_parquet():
        nonlocal buf_id, buf_code, buf_bytes, buf_path
        if not buf_id:
            return
        arr_id = pa.array(buf_id, type=pa.string())
        arr_code = pa.array(buf_code, type=pa.string())
        arr_img = pa.StructArray.from_arrays(
            [pa.array(buf_bytes, type=pa.binary()),
             pa.array(buf_path, type=pa.string())],
            fields=list(schema.field("image").type)
        )
        table = pa.Table.from_arrays([arr_id, arr_img, arr_code], schema=schema)
        writer.write_table(table)
        buf_id, buf_code, buf_bytes, buf_path = [], [], [], []

    # Count total lines (read twice to avoid loading all into memory)
    total = 0
    with in_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total += 1

    err_writer = ErrorWriter(err_jsonl, fsync_every=args.fsync_every)

    try:
        with tempfile.TemporaryDirectory(prefix=f"compile_{idx}_", dir=str(base)) as tmpd:
            tmp_root = Path(tmpd)

            with in_jsonl.open("r", encoding="utf-8", errors="ignore") as f, \
                 tqdm(total=total, desc=f"Split {idx}", ncols=120) as pbar:

                for line in f:
                    line = line.strip()
                    if not line:
                        pbar.update(1)
                        continue

                    # Parse JSON
                    try:
                        obj = json.loads(line)
                    except Exception:
                        fail_cnt += 1
                        err_writer.write_obj({"id": None, "error": "Bad JSON line"})
                        pbar.update(1)
                        pbar.set_postfix(ok=ok_cnt, skip=skipped_cnt, err=fail_cnt)
                        continue

                    row_id = obj.get("id", None)
                    status = obj.get("status", None)
                    code = obj.get("code", "")

                    # Optional: only process status=ok
                    if args.only_ok and status is not None and str(status).lower() != "ok":
                        skipped_cnt += 1
                        # If you want these to be recorded as errors, uncomment the next line
                        err_writer.write_obj({"id": row_id, "error": f"Skipped due to status={status}"})
                        pbar.update(1)
                        pbar.set_postfix(ok=ok_cnt, skip=skipped_cnt, err=fail_cnt)
                        continue

                    img_path_str = f"{row_id}.png" if row_id is not None else "unknown.png"

                    try:
                        workdir = tmp_root / (str(row_id) if row_id is not None else "noid")
                        workdir.mkdir(parents=True, exist_ok=True)

                        png_bytes = compile_and_render(
                            code=str(code) if code is not None else "",
                            workdir=workdir,
                            sif=args.sif,
                            shell_escape=args.shell_escape,
                            dpi=args.dpi,
                            timeout_compile=args.timeout_compile,
                            timeout_convert=args.timeout_convert,
                        )

                        if dump_dir:
                            (dump_dir / img_path_str).write_bytes(png_bytes)

                        buf_id.append("" if row_id is None else str(row_id))
                        buf_code.append(str(code) if code is not None else "")
                        buf_bytes.append(png_bytes)
                        buf_path.append(img_path_str)

                        ok_cnt += 1

                        # Clean up workdir
                        for fn in workdir.iterdir():
                            try:
                                fn.unlink()
                            except Exception:
                                pass

                        if len(buf_id) >= args.batch_size:
                            flush_parquet()

                    except Exception as e:
                        fail_cnt += 1
                        err_writer.write_obj({"id": row_id, "error": str(e)[:2000]})

                    pbar.update(1)
                    pbar.set_postfix(ok=ok_cnt, skip=skipped_cnt, err=fail_cnt)

    finally:
        try:
            flush_parquet()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass
        err_writer.close()

    print(f"[split {idx}] input={in_jsonl}")
    print(f"[split {idx}] output={out_parquet}")
    print(f"[split {idx}] ok={ok_cnt} skip={skipped_cnt} fail={fail_cnt}")


if __name__ == "__main__":
    main()
