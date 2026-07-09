#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
import asyncio
from typing import Any, Dict, Optional, Tuple, Set

import pyarrow.parquet as pq
from tqdm import tqdm
from openai import AsyncOpenAI

# =========================
# Global Config
# =========================
MODEL = os.environ.get("MODEL_PATH", "/path/to/qwen3-235B-instruct-model")

TEMPERATURE = 0.0
TOP_P = 0.95
MAX_TOKENS = 8192
MAX_RETRIES = 2
TIMEOUT_SEC = 1800

MAX_CODE_LEN = 10000

SPLIT_DIR = "splits"
DISTILL_DIR = "distill"
STOP_DIR = "stop_files"

# Parse fenced block (only accepts strict single block)
_FENCE_FULL_RE = re.compile(r"^\s*```latex\s*\n(.*?)\n```+\s*$", re.DOTALL | re.IGNORECASE)

_DOCCLASS_RE = re.compile(r"\\documentclass(\[[^\]]*\])?\{([^}]*)\}", re.IGNORECASE)
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}", re.IGNORECASE)
_END_DOC_RE = re.compile(r"\\end\{document\}", re.IGNORECASE)


def proxy_off():
    os.environ["http_proxy"] = ""
    os.environ["https_proxy"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""


proxy_off()


# =========================
# Prompt builder (MINIMAL FIX)
# =========================
def build_prompt_minimal_fix(code: str, error: str) -> str:
    if code is None:
        code = ""
    if not isinstance(code, str):
        code = str(code)

    if error is None:
        error = ""
    if not isinstance(error, str):
        error = str(error)

    truncated = False
    if len(code) > MAX_CODE_LEN:
        code = code[:MAX_CODE_LEN]
        truncated = True
    trunc_note = "\n[Code truncated]\n" if truncated else ""

    prompt = f"""You are a LaTeX/TikZ compilation repair assistant.

You are given:
1) A LaTeX/TikZ source code that FAILED to compile with pdflatex.
2) The compilation error message excerpt.

Your goal:
Make the SMALLEST POSSIBLE set of changes to make it compile successfully under pdflatex.
Do NOT rewrite the diagram. Do NOT refactor. Do NOT "clean up" unless necessary for compilation.
Keep the output visually and semantically identical to the original intent.

------------------------------------------------
STRICT OUTPUT FORMAT (MUST FOLLOW)
------------------------------------------------
- Your ENTIRE response must be exactly ONE fenced LaTeX block:
  First line: ```latex
  Last line:  ```
- No extra text before or after the block.

------------------------------------------------
COMPILATION TARGET (IMPORTANT)
------------------------------------------------
- Target engine: pdflatex.
- Do NOT use fontspec or other xelatex/lualatex-only packages.
- Avoid shell-escape dependencies.
- Prefer standard TeX Live packages. If a package is missing, try to avoid it by replacing with simpler TikZ/LaTeX constructs.

------------------------------------------------
MINIMAL-CHANGE RULES
------------------------------------------------
1) Only change what is necessary to fix the error message.
2) Preserve original structure, coordinates, and drawing commands whenever possible.
3) If the error is "Undefined control sequence":
   - Add the minimal required package(s), or
   - Replace that command with a pdflatex-compatible equivalent.
4) If the code is a fragment (no documentclass/begin/end):
   - Wrap it into a minimal standalone document ONLY IF necessary to compile.
   - If it already contains a full document, keep it.

------------------------------------------------
COMPILATION ERROR EXCERPT
------------------------------------------------
<ERROR_START>
{error}
<ERROR_END>

------------------------------------------------
ORIGINAL CODE (to minimally patch)
------------------------------------------------
<CODE_START>
{code}
<CODE_END>
{trunc_note}
"""
    return prompt


# =========================
# Output parsing & validation
# =========================
def parse_single_fenced_latex(text: str) -> Optional[str]:
    """Accept only one strict fenced latex block with no extra text outside."""
    if not text:
        return None
    text = text.strip()
    m = _FENCE_FULL_RE.match(text)
    if not m:
        return None
    code = m.group(1).strip()
    return code if code else None


def looks_like_complete_doc(code: str) -> bool:
    if not code:
        return False
    return bool(_DOCCLASS_RE.search(code) and _BEGIN_DOC_RE.search(code) and _END_DOC_RE.search(code))


# =========================
# Dedup (by output jsonl id)
# =========================
def load_existing_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("id", None)
                if rid is not None:
                    ids.add(str(rid))
            except Exception:
                continue
    return ids


# =========================
# Parquet streaming
# =========================
def iter_parquet_records(path: str, existing_ids: Set[str], batch_size: int = 4096):
    """
    Yield records from parquet: {"id":..., "error":..., "code":...}
    Uses ParquetFile.iter_batches to avoid nested conversion issues.
    """
    pf = pq.ParquetFile(path)
    want_cols = ["id", "error", "code"]
    schema_names = set(pf.schema_arrow.names)
    for c in want_cols:
        if c not in schema_names:
            raise KeyError(f"Parquet missing column: {c} in {path}")

    for batch in pf.iter_batches(batch_size=batch_size, columns=want_cols):
        for r in batch.to_pylist():
            rid = r.get("id", None)
            if rid is None:
                continue
            rid = str(rid)
            if not rid:
                continue
            if rid in existing_ids:
                continue
            yield r


# =========================
# Single-sample inference
# =========================
async def generate_fixed_code(item: Dict[str, Any], client: AsyncOpenAI) -> Tuple[str, Optional[str], str]:
    """
    Return (id, fixed_code or None, status)
      status:
        - ok
        - bad_output_format
        - not_complete_doc
        - api_error:...
    """
    item_id = str(item.get("id", "unknown"))
    code = item.get("code", "") or ""
    err = item.get("error", "") or ""

    if not isinstance(code, str):
        code = str(code)
    if not isinstance(err, str):
        err = str(err)

    prompt = build_prompt_minimal_fix(code=code, error=err)

    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                await asyncio.sleep(2 * attempt)

            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT_SEC,
            )

            text = (resp.choices[0].message.content or "").strip()
            fixed = parse_single_fenced_latex(text)
            if fixed is None:
                return item_id, None, "bad_output_format"

            # By default, still require complete document; if you want to allow fragments, relax this check
            if not looks_like_complete_doc(fixed):
                return item_id, None, "not_complete_doc"

            return item_id, fixed, "ok"

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue

    return item_id, None, f"api_error:{last_err[:200]}"


# =========================
# Main async pipeline
# =========================
async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="split index, e.g. 0..9")
    parser.add_argument("--url", type=str, default="127.0.0.1", help="local vllm/serve ip")
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--min-remaining", type=int, default=10, help="if remaining < this, write stop flag and exit")
    parser.add_argument("--batch-size", type=int, default=4096, help="parquet iter batch size")

    parser.add_argument("--input-parquet", type=str, default="", help="override input parquet path")
    parser.add_argument("--output-jsonl", type=str, default="", help="override output jsonl path")

    args = parser.parse_args()

    # Input: splits/{i}.parquet (you have split 10.parquet into 0-9.parquet)
    input_file = args.input_parquet or os.path.join(SPLIT_DIR, f"{args.index}.parquet")
    # Output: distill/{i}.jsonl (you can also change to splits/rewrite_{i}.jsonl)
    output_file = args.output_jsonl or os.path.join(DISTILL_DIR, f"{args.index}.jsonl")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Split parquet file not found: {input_file}")

    existing_ids = load_existing_ids(output_file)
    print(f"[Index {args.index}] already processed ids: {len(existing_ids)}")
    print(f"[Index {args.index}] input parquet: {input_file}")
    print(f"[Index {args.index}] output jsonl:  {output_file}")

    # Estimate remaining
    t0 = time.time()
    remaining = 0
    for _ in iter_parquet_records(input_file, existing_ids, batch_size=args.batch_size):
        remaining += 1
    t1 = time.time()
    print(f"[Index {args.index}] remaining records after dedup: {remaining}, scan_time={t1-t0:.2f}s")

    if remaining < args.min_remaining:
        print(f"Remaining < {args.min_remaining}, create stop flag and exit.")
        stop_file_path = os.path.join(STOP_DIR, f"distill_{args.index}.flag")
        os.makedirs(os.path.dirname(stop_file_path), exist_ok=True)
        with open(stop_file_path, "w", encoding="utf-8") as f:
            f.write("stop\n")
        return

    # OpenAI-compatible client (local server)
    client = AsyncOpenAI(
        base_url=f"http://{args.url}:8000/v1",
        api_key="EMPTY",
    )

    q: asyncio.Queue = asyncio.Queue(maxsize=args.max_concurrent * 4)
    write_lock = asyncio.Lock()

    ok = 0
    fail = 0
    pbar = tqdm(total=remaining, desc="Fixing", ncols=110)

    async def writer(obj: Dict[str, Any]):
        async with write_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    async def worker(worker_id: int):
        nonlocal ok, fail
        while True:
            item = await q.get()
            if item is None:
                q.task_done()
                break

            rid, fixed, status = await generate_fixed_code(item, client)

            if fixed is not None:
                out_obj = {
                    "id": str(rid),         # Remove this line if you want to drop id
                    "status": "ok",
                    "code": fixed,
                }
                await writer(out_obj)
                ok += 1
            else:
                out_obj = {
                    "id": str(rid),         # Same as above
                    "status": status,
                    "code": "",             # Write empty on failure for downstream statistics/secondary processing
                }
                await writer(out_obj)
                fail += 1

            pbar.update(1)
            pbar.set_postfix({"ok": ok, "fail": fail})
            q.task_done()

    workers = [asyncio.create_task(worker(i)) for i in range(args.max_concurrent)]

    for item in iter_parquet_records(input_file, existing_ids, batch_size=args.batch_size):
        await q.put(item)

    for _ in workers:
        await q.put(None)

    await q.join()
    for w in workers:
        await w

    pbar.close()
    print(f"\nDone. Success={ok}, Failed={fail}")
    print(f"Output: {output_file}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
