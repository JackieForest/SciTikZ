import json
import re
from pathlib import Path
from typing import List, Dict, Set
import math
from PIL import Image
from datasets import load_dataset

# utils.py
from pathlib import Path
import json

def load_jsonl_records(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_remaining_records(input_jsonl, output_jsonl=None, prompt_template=None,
                           use_image_basename=False, start_idx=None, end_idx=None):
    """
    - input_jsonl: can be:
        * HF dataset id string like 'user/dataset' -> use datasets.load_dataset(...)
        * local jsonl file path
    - start_idx, end_idx: [start_idx, end_idx) half-open slice; can be None
    Returns list of records (dicts) restricted to [start_idx, end_idx)
    """
    # lazy import
    from pathlib import Path
    from datasets import load_dataset
    input_s = str(input_jsonl)
    records = []

    if ("/" in input_s) and (not Path(input_s).exists()):
        # treat as HF dataset id
        ds = load_dataset(input_s, split="train")
        total = len(ds)
        s = 0 if start_idx is None else start_idx
        e = total if end_idx is None else end_idx
        if s < 0: s = 0
        if e > total: e = total
        if s >= e:
            return []
        # select slice (this loads the slice)
        sel = list(range(s, e))
        sub = ds.select(sel)
        # convert to list of dicts
        records = [dict(r) for r in sub]
    else:
        # treat as local jsonl file
        p = Path(input_s)
        if p.is_dir():
            # try file with .jsonl
            cand = p.with_suffix(".jsonl")
            if cand.exists():
                p = cand
            else:
                raise FileNotFoundError(f"No jsonl found at {p}")
        if not p.exists():
            raise FileNotFoundError(f"No such file: {p}")
        # read lines and slice
        if start_idx is None and end_idx is None:
            # read all
            records = [r for r in load_jsonl_records(p)]
        else:
            s = 0 if start_idx is None else start_idx
            e = None if end_idx is None else end_idx
            # iterate with index to avoid loading full file if large
            records = []
            for i, r in enumerate(load_jsonl_records(p)):
                if i < s:
                    continue
                if e is not None and i >= e:
                    break
                records.append(r)

    # If output_jsonl already exists, you might want to skip already processed records.
    # Your original implementation probably handled that; integrate that logic here if needed.

    return records

def load_image(image_path: str, min_pixels: int=262144, max_pixels: int=1003520) -> Image.Image:
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Resize if too large or too small
        if (image.width * image.height) > max_pixels:
            resize_factor = math.sqrt(max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        if (image.width * image.height) < min_pixels:
            resize_factor = math.sqrt(min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None