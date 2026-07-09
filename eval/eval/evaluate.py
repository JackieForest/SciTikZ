#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from image_metrics import ImageMetrics
from code_metrics import CodeMetrics, extract_document_body, normalize_tex  # only for missing-tex defaults


def list_ids_from_dir(img_dir: Path, suffix: str) -> List[str]:
    ids = []
    for p in sorted(img_dir.glob(f"*{suffix}")):
        if p.is_file():
            ids.append(p.stem)
    return ids


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _safe_getattr(obj, name: str, default):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _finite_or(default: float, x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return float(default)
    return x if math.isfinite(x) else float(default)


def _to_float_or_nan(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_img_dir", type=str, required=True)
    ap.add_argument("--gt_tex_dir", type=str, required=True)
    ap.add_argument("--pred_img_dir", type=str, required=True)
    ap.add_argument("--pred_tex_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--siglip_model", type=str, default="")
    ap.add_argument("--clip_model", type=str, default="")
    ap.add_argument("--dreamsim_model", type=str, default="")  # e.g. "ensemble"; empty string disables

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lpips_net", type=str, default="alex")
    ap.add_argument("--lpips_tau", type=float, default=0.5)
    ap.add_argument("--ssim_resize", type=int, default=384)
    ap.add_argument("--trivial_top_k", type=int, default=500)

    args = ap.parse_args()

    gt_img_dir = Path(args.gt_img_dir)
    gt_tex_dir = Path(args.gt_tex_dir)
    pred_img_dir = Path(args.pred_img_dir)
    pred_tex_dir = Path(args.pred_tex_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert gt_img_dir.exists(), f"GT img dir not found: {gt_img_dir}"
    assert gt_tex_dir.exists(), f"GT tex dir not found: {gt_tex_dir}"
    assert pred_tex_dir.exists(), f"Pred tex dir not found: {pred_tex_dir}"
    # pred_img_dir may not exist/be empty (represents all failures), allowed

    # Use GT images as the reference set
    gt_ids = list_ids_from_dir(gt_img_dir, suffix=".png")
    if not gt_ids:
        raise RuntimeError(f"No GT images found in: {gt_img_dir}")

    gt_set = set(gt_ids)

    pred_img_ids = set(list_ids_from_dir(pred_img_dir, suffix=".png")) if pred_img_dir.exists() else set()
    pred_tex_ids = set([p.stem for p in pred_tex_dir.glob("*.tex")]) if pred_tex_dir.exists() else set()

    missing_pred_img = sorted(list(gt_set - pred_img_ids))
    missing_pred_tex = sorted(list(gt_set - pred_tex_ids))

    (out_dir / "missing_pred_images.txt").write_text("\n".join(missing_pred_img) + "\n", encoding="utf-8")
    (out_dir / "missing_pred_tex.txt").write_text("\n".join(missing_pred_tex) + "\n", encoding="utf-8")

    # --------------------------
    # Image metrics: only computed for ids with pred png
    # --------------------------
    img_metric = ImageMetrics(
        device=args.device,
        siglip_model_path=args.siglip_model if args.siglip_model else None,
        clip_model_path=args.clip_model if args.clip_model else None,
        lpips_net=args.lpips_net,
        lpips_tau=args.lpips_tau,
        ssim_resize=args.ssim_resize,
        fp16=True,
        dreamsim_model_name=args.dreamsim_model if args.dreamsim_model else None,
    )

    img_pairs: List[Tuple[str, str, str]] = []  # (id, gt_png, pred_png)
    for sid in gt_ids:
        gt_png = gt_img_dir / f"{sid}.png"
        pr_png = pred_img_dir / f"{sid}.png"
        if pr_png.exists():
            img_pairs.append((sid, str(gt_png), str(pr_png)))

    pair_list = [(a, b) for _, a, b in img_pairs]
    img_res_map = img_metric.compute_batch(pair_list, batch_size=args.batch_size)

    id_to_img: Dict[str, object] = {}
    for sid, g, p in img_pairs:
        id_to_img[sid] = img_res_map.get((g, p), None)

    # --------------------------
    # Code metrics: computed for all GT ids (but summary only aggregates successful samples)
    # --------------------------
    gt_texts = [read_text(gt_tex_dir / f"{sid}.tex") for sid in gt_ids]

    try:
        code_metric = CodeMetrics(
            gt_corpus_for_crystalbleu=gt_texts,
            crystal_k=args.trivial_top_k,
        )
    except TypeError:
        code_metric = CodeMetrics(trivial_top_k=args.trivial_top_k)  # type: ignore
        if hasattr(code_metric, "build_crystalbleu_trivial"):
            code_metric.build_crystalbleu_trivial(gt_texts)  # type: ignore

    # --------------------------
    # Aggregate per-sample
    # --------------------------
    rows = []
    token_method_counter: Dict[str, int] = {}

    for sid in tqdm(gt_ids, desc="Scoring", ncols=120):
        pred_png_path = pred_img_dir / f"{sid}.png"
        pred_tex_path = pred_tex_dir / f"{sid}.tex"

        has_img = pred_png_path.exists()
        has_pred_tex = pred_tex_path.exists()

        # --------------------------
        # image metrics:
        #   - if has_img but metric missing => NaN (so compiled-only mean skips)
        #   - if no img => NaN (so compiled-only mean skips)
        # --------------------------
        siglip_sim = float("nan")
        clip_sim = float("nan")
        ssim = float("nan")
        siglip_cos = float("nan")
        clip_cos = float("nan")
        lpips_dist = float("nan")
        dreamsim_dist = float("nan")
        lpips_sim = float("nan")
        dreamsim_sim = float("nan")

        if has_img:
            r = id_to_img.get(sid, None)
            if r is not None:
                siglip_sim = _to_float_or_nan(_safe_getattr(r, "siglip_sim", float("nan")))
                clip_sim = _to_float_or_nan(_safe_getattr(r, "clip_sim", float("nan")))
                ssim = _to_float_or_nan(_safe_getattr(r, "ssim", float("nan")))

                siglip_cos = _to_float_or_nan(_safe_getattr(r, "siglip_cos", float("nan")))
                clip_cos = _to_float_or_nan(_safe_getattr(r, "clip_cos", float("nan")))

                lpips_sim = _to_float_or_nan(_safe_getattr(r, "lpips_sim", float("nan")))
                lpips_dist = _to_float_or_nan(_safe_getattr(r, "lpips_dist", float("nan")))

                dreamsim_sim = _to_float_or_nan(_safe_getattr(r, "dreamsim_sim", float("nan")))
                dreamsim_dist = _to_float_or_nan(_safe_getattr(r, "dreamsim_dist", float("nan")))

        # --------------------------
        # code metrics:
        #   - if missing pred tex => NaN for sims (skip in compiled-only mean),
        #     but keep token_edit_dist numeric for debugging if you want
        # --------------------------
        gt_tex = read_text(gt_tex_dir / f"{sid}.tex")
        pred_tex = read_text(pred_tex_path) if has_pred_tex else ""

        cm = code_metric.compute_one(gt_tex, pred_tex) if pred_tex else None  # type: ignore

        token_dist = float("nan")
        token_dist_norm = float("nan")
        token_sim = float("nan")
        token_method = "missing_pred_tex"
        crystalbleu = float("nan")

        if cm is not None:
            token_dist = _to_float_or_nan(_safe_getattr(cm, "token_edit_dist", _safe_getattr(cm, "ted_dist", float("nan"))))
            token_sim = _to_float_or_nan(_safe_getattr(cm, "token_edit_sim", _safe_getattr(cm, "ted_sim", float("nan"))))
            token_dist_norm = _to_float_or_nan(_safe_getattr(cm, "token_edit_dist_norm", float("nan")))
            token_method = str(_safe_getattr(cm, "ted_method", "token_edit_distance"))
            crystalbleu = _to_float_or_nan(_safe_getattr(cm, "crystalbleu", float("nan")))
        else:
            # Optional: still assign a "reference length" to token_dist for debugging (but summary doesn't use it)
            try:
                gt_body = normalize_tex(extract_document_body(gt_tex))
                ref_tokens = code_metric.token_edit_metric.tokenize_to_tokens(  # type: ignore
                    gt_body,
                    language=code_metric.token_edit_metric.language,          # type: ignore
                )
                ref_len = max(len(ref_tokens), 1)
                token_dist = float(ref_len)
            except Exception:
                token_dist = float("nan")

        token_method_counter[token_method] = token_method_counter.get(token_method, 0) + 1

        rows.append({
            "id": sid,
            "has_pred_image": int(has_img),
            "has_pred_tex": int(has_pred_tex),

            # native outputs
            "siglip_cos": siglip_cos,
            "clip_cos": clip_cos,
            "lpips_dist": lpips_dist,
            "dreamsim_dist": dreamsim_dist,
            "token_edit_dist_norm": token_dist_norm,

            # sims
            "siglip_sim": siglip_sim,
            "clip_sim": clip_sim,
            "lpips_sim": lpips_sim,
            "ssim": ssim,
            "dreamsim_sim": dreamsim_sim,

            # code side
            "token_edit_dist": token_dist,
            "token_edit_sim": token_sim,
            "token_edit_method": token_method,
            "crystalbleu": crystalbleu,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sample.csv", index=False)

    # --------------------------
    # summary
    #   - compile_success_rate: ALL samples
    #   - other metrics: ONLY compiled-success samples
    # --------------------------
    n = len(df)
    n_success = int(df["has_pred_image"].sum())
    n_has_tex = int(df["has_pred_tex"].sum())
    compile_success_rate = float(n_success / max(n, 1))
    tex_available_rate = float(n_has_tex / max(n, 1))

    df_succ = df[df["has_pred_image"] == 1].copy()
    df_has_tex = df[df["has_pred_tex"] == 1].copy()

    def mean_on_success(col: str) -> float:
        # pandas automatically skips NaN
        if df_succ.empty:
            return float("nan")
        return float(pd.to_numeric(df_succ[col], errors="coerce").mean())

    def mean_on_has_tex(col: str) -> float:
        # pandas automatically skips NaN
        if df_has_tex.empty:
            return float("nan")
        return float(pd.to_numeric(df_has_tex[col], errors="coerce").mean())

    # Additional: if you want to keep "failure counts as worst" overall mean, here's one (can be deleted)
    # Here follows your original policy: siglip/clip/ssim missing->0, lpips/dreamsim sim missing->0
    def mean_overall_with_worst_defaults(col: str, default: float) -> float:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.fillna(default)
        return float(s.mean())

    summary = {
        "gt_count": int(n),
        "pred_image_count": int(n_success),
        "pred_tex_count": int(n_has_tex),
        "compile_success_rate": compile_success_rate,
        "tex_available_rate": tex_available_rate,

        # ========== compiled-only means (your main metric scope) ==========
        "compiled_only": {
            "count": int(n_success),

            "mean_siglip_cos": mean_on_success("siglip_cos"),
            "mean_clip_cos": mean_on_success("clip_cos"),
            "mean_lpips_dist": mean_on_success("lpips_dist"),
            "mean_dreamsim_dist": mean_on_success("dreamsim_dist"),

            "mean_siglip_sim": mean_on_success("siglip_sim"),
            "mean_clip_sim": mean_on_success("clip_sim"),
            "mean_lpips_sim": mean_on_success("lpips_sim"),
            "mean_ssim": mean_on_success("ssim"),
            "mean_dreamsim_sim": mean_on_success("dreamsim_sim"),

            # code side: still only counted in the "has image success" set
            # (if you want code metrics denominator to be has_pred_tex==1, can split further)
            "mean_token_edit_dist_norm": mean_on_success("token_edit_dist_norm"),
            "mean_token_edit_dist": mean_on_success("token_edit_dist"),
            "mean_token_edit_sim": mean_on_success("token_edit_sim"),
            "mean_crystalbleu": mean_on_success("crystalbleu"),
        },

        # ========== code metrics on all samples with tex (independent of image rendering) ==========
        "code_metrics_all_tex": {
            "count": int(n_has_tex),
            "mean_token_edit_dist_norm": mean_on_has_tex("token_edit_dist_norm"),
            "mean_token_edit_dist": mean_on_has_tex("token_edit_dist"),
            "mean_token_edit_sim": mean_on_has_tex("token_edit_sim"),
            "mean_crystalbleu": mean_on_has_tex("crystalbleu"),
        },

        # ========== optional: overall means (failure counts as worst, for comparing with old tables) ==========
        "overall_with_worst_defaults": {
            "mean_siglip_sim": mean_overall_with_worst_defaults("siglip_sim", 0.0),
            "mean_clip_sim": mean_overall_with_worst_defaults("clip_sim", 0.0),
            "mean_lpips_dist": mean_overall_with_worst_defaults("lpips_dist", 1.0),
            "mean_dreamsim_dist": mean_overall_with_worst_defaults("dreamsim_dist", 1.0),
            "mean_ssim": mean_overall_with_worst_defaults("ssim", 0.0),
            
            "mean_lpips_sim": mean_overall_with_worst_defaults("lpips_sim", 0.0),
            "mean_dreamsim_sim": mean_overall_with_worst_defaults("dreamsim_sim", 0.0),
            "mean_token_edit_sim": mean_overall_with_worst_defaults("token_edit_sim", 0.0),
            "mean_crystalbleu": mean_overall_with_worst_defaults("crystalbleu", 0.0),
        },

        "token_edit_method_counter": token_method_counter,
        "missing_pred_images": int(len(missing_pred_img)),
        "missing_pred_tex": int(len(missing_pred_tex)),
        "dreamsim_enabled": bool(args.dreamsim_model),

        "metric_policy": {
            "compile_success_rate_denominator": "ALL(gt_ids)",
            "image_metrics_denominator": "ONLY(has_pred_image==1)",
            "code_metrics_denominator": "ONLY(has_pred_tex==1) for code_metrics_all_tex, ONLY(has_pred_image==1) for compiled_only",
            "missing_metric_value": "NaN (excluded from mean)",
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("==== DONE ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
