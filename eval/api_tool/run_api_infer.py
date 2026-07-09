#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# pip install openai
from openai import OpenAI


LATEX_BLOCK_RE = re.compile(r"```latex\s*([\s\S]*?)\s*```", re.IGNORECASE)


def guess_mime(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".jpg") or p.endswith(".jpeg"):
        return "image/jpeg"
    if p.endswith(".webp"):
        return "image/webp"
    return "image/png"


def image_to_data_url(image_path: str) -> str:
    mime = guess_mime(image_path)
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def strip_image_token(question: str) -> str:
    q = question.strip()
    q = q.replace("<image>", "").strip()
    return q


def parse_first_latex_block(text: str) -> Optional[str]:
    if text is None:
        return None
    m = LATEX_BLOCK_RE.search(text)
    if not m:
        return None
    return (m.group(1) or "").strip("\n")


def enforce_single_latex_block(text: str) -> str:
    if text is None:
        text = ""
    
    text = text.strip()
    if not text:
        return "```latex\n\n```"
    
    while True:
        if text.startswith("```latex\n"):
            text = text[10:]
        elif text.startswith("```latex"):
            text = text[8:]
        else:
            break
        text = text.lstrip()
    
    closing_marker_idx = text.find("```")
    
    if closing_marker_idx > 0:
        body = text[:closing_marker_idx].strip()
    else:
        body = text.strip()
    
    body = re.sub(r"```(?:latex)?\s*", "", body, flags=re.IGNORECASE)
    body = body.strip()
    
    return f"```latex\n{body}\n```"


def build_messages(image_path: str, question: str) -> List[Dict[str, Any]]:
    q = strip_image_token(question)
    data_url = image_to_data_url(image_path)

    system = (
        "You are an expert LaTeX developer who specializes in creating scientific Graphics. "
        "Generate precise, well-structured TikZ/LaTeX code to faithfully recreate the image. "
        "The code must be complete and compilable."
    )

    user_text = (
        f"{q}\n\n"
        "CRITICAL OUTPUT REQUIREMENT:\n"
        "- Output ONLY one code block fenced by ```latex and ```.\n"
        "- No surrounding text, no explanations, no commentary.\n"
        "- The code must be **complete** and **compilable**."
    )

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def call_with_retry(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    timeout_sec: float,
    max_retries: int,
) -> Tuple[str, Optional[str]]:
    """
    Returns (output_text, error_message)
    """
    backoff = 2.0
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            # Prepare parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout_sec,
            }

            resp = client.chat.completions.create(**params)
            
            # Check if response has choices
            if not resp.choices or len(resp.choices) == 0:
                error_message = f"Model {model} returned no choices in response"
                print(f"ERROR: {error_message}")
                return "```latex\n\n```", error_message
            
            # Get content from response
            choice = resp.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            text = getattr(choice.message, "content", None) or ""
            
            # Check finish_reason for potential issues
            if finish_reason == "length":
                print(f"WARN: Model {model} response was truncated (finish_reason=length)")
            elif finish_reason == "content_filter":
                print(f"WARN: Model {model} response was filtered (finish_reason=content_filter)")
            
            # If the response is empty, log detailed info
            if not text or not text.strip():
                # Get image info for debugging (truncate long base64 strings)
                image_url = messages[1]['content'][0]['image_url']['url']
                if len(image_url) > 200:
                    image_info = f"{image_url[:100]}...{image_url[-50:]}"
                else:
                    image_info = image_url
                error_message = f"Model {model} returned empty response (finish_reason={finish_reason})"
                print(f"ERROR: {error_message}")
                print(f"DEBUG: Image path: {messages[1]['content'][1]['text'][:100]}...")
                return "```latex\n\n```", error_message

            return text, None
        except Exception as e:
            last_err = repr(e)
            if attempt >= max_retries:
                break
            print(f"Attempt {attempt + 1} failed: {last_err}")
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 20.0)
    return "", last_err


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj.get("id"))
            except Exception:
                continue
    return done


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name)


def list_models_safe(client: OpenAI) -> List[str]:
    try:
        ms = client.models.list()
        data = getattr(ms, "data", []) or []
        ids = []
        for m in data:
            mid = getattr(m, "id", None)
            if mid:
                ids.append(mid)
        return ids
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl path (ours.jsonl)")
    ap.add_argument("--out_dir", required=True, help="Output directory for per-model jsonl")
    ap.add_argument("--base_url", default=os.environ.get("BASE_URL", ""), help="Proxy base_url")
    ap.add_argument("--api_key", default=os.environ.get("API_KEY", ""), help="Proxy api_key")
    ap.add_argument("--models", required=True, help="Comma-separated model names")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--prompt_name", default="parse")
    args = ap.parse_args()

    if not args.base_url:
        raise SystemExit("ERROR: base_url is empty. Provide --base_url or set BASE_URL env.")
    if not args.api_key:
        raise SystemExit("ERROR: api_key is empty. Provide --api_key or set API_KEY env.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("ERROR: no models provided.")

    data = load_jsonl(args.input)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    visible_models = list_models_safe(client)
    if not visible_models:
        raise SystemExit(
            "ERROR: /v1/models returned empty list for this API_KEY. "
            "Likely your key/group has no enabled distributors/models. "
            "Use an AUTO group key or ask admin to enable models."
        )

    for model in models:
        safe_name = sanitize_filename(model)
        out_path = out_dir / f"{safe_name}.jsonl"

        done_ids = load_done_ids(out_path)
        todo = [x for x in data if x.get("id") not in done_ids]

        print(f"\n=== Model: {model} ===")
        print(f"[OUT] {out_path}")
        print(f"[RESUME] done={len(done_ids)} todo={len(todo)} total={len(data)}")

        # Thread pool for IO-bound API calls
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = []
            for item in todo:
                image_path = item["image_path"]
                question = item["question"]
                messages = build_messages(image_path, question)

                futures.append(
                    ex.submit(
                        call_with_retry,
                        client,
                        model,
                        messages,
                        args.temperature,
                        args.max_tokens,
                        args.top_p,
                        args.timeout,
                        args.retries,
                    )
                )

            # Write incrementally (append)
            with out_path.open("a", encoding="utf-8") as wf:
                for item, fut in tqdm(list(zip(todo, futures)), total=len(todo), desc=f"{model}", ncols=100):
                    raw_text, err = fut.result()
                    final_text = enforce_single_latex_block(raw_text)

                    out_obj = {
                        "id": item["id"],
                        "image_path": item["image_path"],
                        "question": item["question"],
                        "outputs": [final_text],
                        "prompt_name": args.prompt_name,
                        "model": model,
                        "gen": {
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens,
                        },
                    }
                    if err is not None:
                        out_obj["error"] = err
                        out_obj["outputs"] = ["```latex\n\n```"]

                    wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    wf.flush()

        print(f"[DONE] Wrote: {out_path}")


if __name__ == "__main__":
    main()
