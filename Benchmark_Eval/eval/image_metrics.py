#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal
import math

from tqdm import tqdm

import torch
from PIL import Image, ImageChops

import torchvision.transforms as T

# SSIM
try:
    from pytorch_msssim import ssim as ssim_fn
except Exception:
    ssim_fn = None

# LPIPS
try:
    import lpips
except Exception:
    lpips = None

# DreamSim
try:
    from dreamsim import dreamsim as dreamsim_fn
except Exception:
    dreamsim_fn = None

try:
    from huggingface_hub import cached_assets_path
except Exception:
    cached_assets_path = None

# Transformers (SigLIP / CLIP)
from transformers import CLIPProcessor, CLIPModel
from transformers import SiglipModel, SiglipImageProcessor


@dataclass
class ImageMetricResult:
    siglip_sim: float = 0.0
    clip_sim: float = 0.0

    siglip_cos: float = float("nan")   # cosine in [-1,1] (after L2 norm)
    clip_cos: float = float("nan")

    lpips_dist: float = float("nan")
    lpips_sim: float = 0.0

    ssim: float = 0.0

    dreamsim_dist: float = float("nan")  # native distance (lower=better; impl-dependent)
    dreamsim_sim: float = 0.0


def _safe_open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _cosine_to_01(cos: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1] (your code uses clamp to [0,1] for stability)
    return cos.clamp(min=0.0, max=1.0)


def _trim_white_border(
    img: Image.Image,
    *,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    pad: int = 2,
) -> Image.Image:
    """
    Trim near-background border by computing bbox of difference w.r.t. a solid bg.
    Works well for TikZ renders that have white margins.

    pad: expand bbox by pad pixels (keeps a small margin after trimming).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    bg = Image.new("RGB", img.size, bg_color)
    diff = ImageChops.difference(img, bg).convert("L")
    bbox = diff.getbbox()
    if bbox is None:
        return img  # fully background

    left, upper, right, lower = bbox
    left = max(0, left - pad)
    upper = max(0, upper - pad)
    right = min(img.width, right + pad)
    lower = min(img.height, lower + pad)
    return img.crop((left, upper, right, lower))


def _pad_to_size(
    img: Image.Image,
    target_w: int,
    target_h: int,
    *,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Center-pad an image to (target_w, target_h) with a solid background."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width == target_w and img.height == target_h:
        return img
    out = Image.new("RGB", (target_w, target_h), fill)
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    out.paste(img, (x, y))
    return out


def _center_crop(
    img: Image.Image,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """Center-crop an image to (target_w, target_h)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width == target_w and img.height == target_h:
        return img
    if target_w > img.width or target_h > img.height:
        # If requested crop is bigger, fall back to padding (safer than raising)
        return _pad_to_size(img, max(target_w, img.width), max(target_h, img.height))
    left = (img.width - target_w) // 2
    top = (img.height - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _resize_max_side(
    img: Image.Image,
    max_side: int,
) -> Image.Image:
    """Downscale (only) to ensure max(width,height) <= max_side, preserving aspect."""
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def _align_pair_for_ssim(
    gt_img: Image.Image,
    pr_img: Image.Image,
    *,
    trim: bool,
    trim_pad: int,
    align: Literal["pad", "crop"],
    bg_color: Tuple[int, int, int],
    max_side: int,
) -> Tuple[Image.Image, Image.Image]:
    """
    Align a (gt, pred) pair to the same size for SSIM.

    Recommended default for TikZ:
      trim=True + align="pad"
    This reduces border-induced penalties while preserving content.
    """
    if trim:
        gt_img = _trim_white_border(gt_img, bg_color=bg_color, pad=trim_pad)
        pr_img = _trim_white_border(pr_img, bg_color=bg_color, pad=trim_pad)

    # Optional downscale to cap compute (apply before final size match)
    if max_side and max_side > 0:
        gt_img = _resize_max_side(gt_img, max_side)
        pr_img = _resize_max_side(pr_img, max_side)

    gw, gh = gt_img.size
    pw, ph = pr_img.size

    if align == "pad":
        tw, th = max(gw, pw), max(gh, ph)
        gt_img = _pad_to_size(gt_img, tw, th, fill=bg_color)
        pr_img = _pad_to_size(pr_img, tw, th, fill=bg_color)
        return gt_img, pr_img

    # align == "crop"
    tw, th = min(gw, pw), min(gh, ph)
    tw = max(1, tw)
    th = max(1, th)
    gt_img = _center_crop(gt_img, tw, th)
    pr_img = _center_crop(pr_img, tw, th)
    return gt_img, pr_img


class ImageMetrics:
    """
    Compute: SigLIP / CLIP / LPIPS / SSIM / DreamSim

    Outputs native/raw values:
    - SigLIP & CLIP: cosine similarity in [-1,1] (siglip_cos/clip_cos), also provides sim in [0,1] (siglip_sim/clip_sim)
    - LPIPS: distance value (lpips_dist), also provides similarity via exp(-dist/tau) (lpips_sim)
    - SSIM: raw SSIM score in [0,1] from pytorch_msssim.ssim
    - DreamSim: native distance value (dreamsim_dist), also provides similarity via 1-dist (dreamsim_sim)

    Key implementation details:
    1) SigLIP/CLIP: Concatenate (gt_imgs + pr_imgs) in each batch for a single forward pass (closer to testsiglip path)
    2) SigLIP/CLIP: Force float32 for feature extraction by default (avoid fp16 autocast causing artificially high/stepped similarity)
    """

    def __init__(
        self,
        device: str = "cuda",
        siglip_model_path: Optional[str] = None,
        clip_model_path: Optional[str] = None,
        lpips_net: str = "alex",
        lpips_tau: float = 0.5,
        lpips_resize: int = 384,
        # --- SSIM options ---
        ssim_resize: Optional[int] = None,  # None => do not force resize
        ssim_align: Literal["pad", "crop"] = "pad",
        ssim_trim_border: bool = True,
        ssim_trim_pad: int = 2,
        ssim_bg_color: Tuple[int, int, int] = (255, 255, 255),
        ssim_max_side: int = 1024,  # cap compute; 0/None disables
        fp16: bool = True,         
        # --- SigLIP/CLIP embedding options ---
        embed_fp32: bool = True,     # SigLIP/CLIP features default to float32
        embed_one_forward: bool = True,  # SigLIP/CLIP concatenate batch for single forward pass
        # --- DreamSim options ---
        dreamsim_model_name: Optional[str] = None,  # e.g. "ensemble"; None disables
        dreamsim_pretrained: bool = True,
        dreamsim_normalize: bool = True,
        dreamsim_preprocess: bool = True,           # mimic ref: expand(... do_trim=True)
        dreamsim_trim_pad: int = 2,                 # used when dreamsim_preprocess=True
        dreamsim_bg_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.device = torch.device(device)
        self.fp16 = fp16 and (self.device.type == "cuda")
        self.embed_fp32 = bool(embed_fp32)
        self.embed_one_forward = bool(embed_one_forward)

        # --- SigLIP ---
        self.siglip_model = None
        self.siglip_proc = None
        if siglip_model_path:
            self.siglip_proc = SiglipImageProcessor.from_pretrained(siglip_model_path)
            self.siglip_model = SiglipModel.from_pretrained(siglip_model_path).to(self.device)
            self.siglip_model.eval()

        # --- CLIP ---
        self.clip_model = None
        self.clip_proc = None
        if clip_model_path:
            self.clip_proc = CLIPProcessor.from_pretrained(clip_model_path)
            self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
            self.clip_model.eval()

        # --- LPIPS ---
        self.lpips_tau = float(lpips_tau)
        self.lpips_model = None
        if lpips is not None:
            self.lpips_model = lpips.LPIPS(net=lpips_net).to(self.device)
            self.lpips_model.eval()

        self.lpips_resize = int(lpips_resize)
        self._lpips_tf = T.Compose([
            T.Resize((self.lpips_resize, self.lpips_resize), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),  # [0,1]
            T.Lambda(lambda x: x * 2.0 - 1.0),  # [-1,1]
        ])

        # --- SSIM ---
        self.ssim_resize = None if (ssim_resize is None) else int(ssim_resize)
        self.ssim_align = ssim_align
        self.ssim_trim_border = bool(ssim_trim_border)
        self.ssim_trim_pad = int(ssim_trim_pad)
        self.ssim_bg_color = tuple(ssim_bg_color)
        self.ssim_max_side = int(ssim_max_side) if ssim_max_side else 0

        self._to_tensor01 = T.ToTensor()  # [0,1]
        if self.ssim_resize is not None:
            self._ssim_resize_tf = T.Resize(
                (self.ssim_resize, self.ssim_resize),
                interpolation=T.InterpolationMode.BICUBIC
            )
        else:
            self._ssim_resize_tf = None

        # --- DreamSim ---
        self.dreamsim_model = None
        self.dreamsim_proc = None
        self.dreamsim_preprocess = bool(dreamsim_preprocess)
        self.dreamsim_trim_pad = int(dreamsim_trim_pad)
        self.dreamsim_bg_color = tuple(dreamsim_bg_color)

        if self.device.type == "cuda":
            try:
                bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            except Exception:
                bf16_ok = False
            self.dreamsim_dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            self.dreamsim_dtype = torch.float32

        if dreamsim_model_name and (dreamsim_fn is not None):
            cache_dir = None
            if cached_assets_path is not None:
                try:
                    cache_dir = str(cached_assets_path(library_name="evaluate", namespace="dreamsim"))
                except Exception:
                    cache_dir = None

            model, processor = dreamsim_fn(
                dreamsim_type=dreamsim_model_name,
                pretrained=dreamsim_pretrained,
                normalize_embeds=dreamsim_normalize,
                device=str(self.device),
                cache_dir=cache_dir,
            )

            try:
                for extractor in getattr(model, "extractor_list", []):
                    if hasattr(extractor, "model"):
                        extractor.model = extractor.model.to(self.dreamsim_dtype)
                    if hasattr(extractor, "proj"):
                        extractor.proj = extractor.proj.to(self.dreamsim_dtype)
            except Exception:
                pass

            self.dreamsim_model = model.to(self.dreamsim_dtype)
            self.dreamsim_proc = processor
            try:
                self.dreamsim_model.eval()
            except Exception:
                pass

    @torch.no_grad()
    def compute_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 16,
    ) -> Dict[Tuple[str, str], ImageMetricResult]:
        """
        pairs: list of (gt_png_path, pred_png_path)
        return: dict keyed by (gt_path, pred_path)
        """
        out: Dict[Tuple[str, str], ImageMetricResult] = {}
        if not pairs:
            return out

        for p in pairs:
            out[p] = ImageMetricResult()

        if self.siglip_model is not None:
            self._compute_siglip(out, pairs, batch_size)

        if self.clip_model is not None:
            self._compute_clip(out, pairs, batch_size)

        if self.lpips_model is not None:
            self._compute_lpips(out, pairs, batch_size)

        if ssim_fn is not None:
            self._compute_ssim(out, pairs, batch_size)

        if self.dreamsim_model is not None and self.dreamsim_proc is not None:
            self._compute_dreamsim(out, pairs, batch_size)

        return out

    def _autocast_ctx(self):
        if self.fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)

        class _NoCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        return _NoCtx()

    # -----------------------
    # SigLIP (modified)
    # -----------------------
    def _compute_siglip(self, out, pairs, batch_size):
        assert self.siglip_proc is not None and self.siglip_model is not None

        for i in tqdm(range(0, len(pairs), batch_size), desc="SigLIP", ncols=120):
            batch = pairs[i:i + batch_size]
            gt_imgs = [_safe_open_rgb(a) for a, _ in batch]
            pr_imgs = [_safe_open_rgb(b) for _, b in batch]

            if self.embed_one_forward:
                # Aligned with testsiglip: both images go through the same forward pass (extended to batch here)
                all_imgs = gt_imgs + pr_imgs
                inputs = self.siglip_proc(images=all_imgs, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    feats = self.siglip_model.get_image_features(**inputs)
                    if self.embed_fp32:
                        feats = feats.float()

                feats = torch.nn.functional.normalize(feats, dim=-1)
                B = len(batch)
                gt_feat = feats[:B]
                pr_feat = feats[B:]
                cos = (gt_feat * pr_feat).sum(dim=-1).clamp(-1, 1)
                cos_list = cos.detach().float().cpu().tolist()
                sim01 = _cosine_to_01(cos).detach().float().cpu().tolist()

            else:
                # Compatibility mode: two separate forward passes (not recommended)
                gt_inputs = self.siglip_proc(images=gt_imgs, return_tensors="pt")
                pr_inputs = self.siglip_proc(images=pr_imgs, return_tensors="pt")
                gt_inputs = {k: v.to(self.device) for k, v in gt_inputs.items()}
                pr_inputs = {k: v.to(self.device) for k, v in pr_inputs.items()}

                with torch.inference_mode():
                    gt_feat = self.siglip_model.get_image_features(**gt_inputs)
                    pr_feat = self.siglip_model.get_image_features(**pr_inputs)
                    if self.embed_fp32:
                        gt_feat = gt_feat.float()
                        pr_feat = pr_feat.float()

                gt_feat = torch.nn.functional.normalize(gt_feat, dim=-1)
                pr_feat = torch.nn.functional.normalize(pr_feat, dim=-1)
                cos = (gt_feat * pr_feat).sum(dim=-1).clamp(-1, 1)
                cos_list = cos.detach().float().cpu().tolist()
                sim01 = _cosine_to_01(cos).detach().float().cpu().tolist()

            for (key, c, s) in zip(batch, cos_list, sim01):
                out[key].siglip_cos = float(c)
                out[key].siglip_sim = float(s)

    # -----------------------
    # CLIP (modified)
    # -----------------------
    def _compute_clip(self, out, pairs, batch_size):
        assert self.clip_proc is not None and self.clip_model is not None

        for i in tqdm(range(0, len(pairs), batch_size), desc="CLIP", ncols=120):
            batch = pairs[i:i + batch_size]
            gt_imgs = [_safe_open_rgb(a) for a, _ in batch]
            pr_imgs = [_safe_open_rgb(b) for _, b in batch]

            if self.embed_one_forward:
                all_imgs = gt_imgs + pr_imgs
                inputs = self.clip_proc(images=all_imgs, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    feats = self.clip_model.get_image_features(**inputs)
                    if self.embed_fp32:
                        feats = feats.float()

                feats = torch.nn.functional.normalize(feats, dim=-1)
                B = len(batch)
                gt_feat = feats[:B]
                pr_feat = feats[B:]
                cos = (gt_feat * pr_feat).sum(dim=-1).clamp(-1, 1)
                cos_list = cos.detach().float().cpu().tolist()
                sim01 = _cosine_to_01(cos).detach().float().cpu().tolist()

            else:
                gt_inputs = self.clip_proc(images=gt_imgs, return_tensors="pt")
                pr_inputs = self.clip_proc(images=pr_imgs, return_tensors="pt")
                gt_inputs = {k: v.to(self.device) for k, v in gt_inputs.items()}
                pr_inputs = {k: v.to(self.device) for k, v in pr_inputs.items()}

                with torch.inference_mode():
                    gt_feat = self.clip_model.get_image_features(**gt_inputs)
                    pr_feat = self.clip_model.get_image_features(**pr_inputs)
                    if self.embed_fp32:
                        gt_feat = gt_feat.float()
                        pr_feat = pr_feat.float()

                gt_feat = torch.nn.functional.normalize(gt_feat, dim=-1)
                pr_feat = torch.nn.functional.normalize(pr_feat, dim=-1)
                cos = (gt_feat * pr_feat).sum(dim=-1).clamp(-1, 1)
                cos_list = cos.detach().float().cpu().tolist()
                sim01 = _cosine_to_01(cos).detach().float().cpu().tolist()

            for (key, c, s) in zip(batch, cos_list, sim01):
                out[key].clip_cos = float(c)
                out[key].clip_sim = float(s)

    def _compute_lpips(self, out, pairs, batch_size):
        assert self.lpips_model is not None

        for i in tqdm(range(0, len(pairs), batch_size), desc="LPIPS", ncols=120):
            batch = pairs[i:i + batch_size]
            gt_t = torch.stack([self._lpips_tf(_safe_open_rgb(a)) for a, _ in batch], dim=0).to(self.device)
            pr_t = torch.stack([self._lpips_tf(_safe_open_rgb(b)) for _, b in batch], dim=0).to(self.device)

            with self._autocast_ctx():
                d = self.lpips_model(gt_t, pr_t)  # [B,1,1,1] or [B,1]
            d = d.view(-1).detach().float().cpu()

            for (key, dist) in zip(batch, d.tolist()):
                dist = float(dist)
                sim = math.exp(-dist / self.lpips_tau) if self.lpips_tau > 0 else 0.0
                out[key].lpips_dist = dist
                out[key].lpips_sim = float(sim)

    def _img_to_ssim_tensor(self, img: Image.Image) -> torch.Tensor:
        if self._ssim_resize_tf is not None:
            img = self._ssim_resize_tf(img)
        t = self._to_tensor01(img)  # (C,H,W) in [0,1]
        return t

    def _compute_ssim(self, out, pairs, batch_size):
        if ssim_fn is None:
            return

        for i in tqdm(range(0, len(pairs), batch_size), desc="SSIM", ncols=120):
            batch = pairs[i:i + batch_size]

            grouped: Dict[Tuple[int, int], List[Tuple[Tuple[str, str], torch.Tensor, torch.Tensor]]] = {}

            for key in batch:
                gt_path, pr_path = key
                gt_img = _safe_open_rgb(gt_path)
                pr_img = _safe_open_rgb(pr_path)

                gt_img, pr_img = _align_pair_for_ssim(
                    gt_img,
                    pr_img,
                    trim=self.ssim_trim_border,
                    trim_pad=self.ssim_trim_pad,
                    align=self.ssim_align,
                    bg_color=self.ssim_bg_color,
                    max_side=self.ssim_max_side,
                )

                gt_t = self._img_to_ssim_tensor(gt_img)
                pr_t = self._img_to_ssim_tensor(pr_img)

                if gt_t.shape[-2:] != pr_t.shape[-2:]:
                    gh, gw = gt_t.shape[-2], gt_t.shape[-1]
                    ph, pw = pr_t.shape[-2], pr_t.shape[-1]
                    th, tw = max(gh, ph), max(gw, pw)

                    def pad_tensor(t: torch.Tensor, th: int, tw: int) -> torch.Tensor:
                        c, h, w = t.shape
                        out_t = torch.ones((c, th, tw), dtype=t.dtype)  # white in [0,1]
                        y = (th - h) // 2
                        x = (tw - w) // 2
                        out_t[:, y:y + h, x:x + w] = t
                        return out_t

                    gt_t = pad_tensor(gt_t, th, tw)
                    pr_t = pad_tensor(pr_t, th, tw)

                shape_key = (gt_t.shape[-2], gt_t.shape[-1])  # (H,W)
                grouped.setdefault(shape_key, []).append((key, gt_t, pr_t))

            for _, items in grouped.items():
                keys = [k for (k, _, _) in items]
                gt_stack = torch.stack([t for (_, t, _) in items], dim=0).to(self.device)
                pr_stack = torch.stack([t for (_, _, t) in items], dim=0).to(self.device)

                s = ssim_fn(gt_stack, pr_stack, data_range=1.0, size_average=False)  # [B]
                s_list = s.detach().float().cpu().tolist()

                for key, val in zip(keys, s_list):
                    out[key].ssim = float(val)

    # -----------------------
    # DreamSim
    # -----------------------
    def _expand_for_dreamsim(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.dreamsim_preprocess:
            img = _trim_white_border(img, bg_color=self.dreamsim_bg_color, pad=self.dreamsim_trim_pad)
            side = max(img.size)
            img = _pad_to_size(img, side, side, fill=self.dreamsim_bg_color)
        return img

    def _compute_dreamsim(self, out, pairs, batch_size):
        assert self.dreamsim_model is not None and self.dreamsim_proc is not None

        for i in tqdm(range(0, len(pairs), batch_size), desc="DreamSim", ncols=120):
            batch = pairs[i:i + batch_size]
            gt_imgs = [self._expand_for_dreamsim(_safe_open_rgb(a)) for a, _ in batch]
            pr_imgs = [self._expand_for_dreamsim(_safe_open_rgb(b)) for _, b in batch]

            def _to_chw(t: torch.Tensor) -> torch.Tensor:
                if not isinstance(t, torch.Tensor):
                    raise TypeError(type(t))
                if t.dim() == 4 and t.shape[0] == 1:
                    return t.squeeze(0)
                if t.dim() == 3:
                    return t
                raise ValueError(f"Unexpected dreamsim_proc output shape: {tuple(t.shape)}")

            try:
                gt_list = [self.dreamsim_proc(im) for im in gt_imgs]
                pr_list = [self.dreamsim_proc(im) for im in pr_imgs]

                if isinstance(gt_list[0], dict) or isinstance(pr_list[0], dict):
                    raise TypeError("dreamsim_proc returned dict; use batched mode")

                gt_t = torch.stack([_to_chw(t) for t in gt_list], dim=0)  # [B,C,H,W]
                pr_t = torch.stack([_to_chw(t) for t in pr_list], dim=0)  # [B,C,H,W]

            except Exception:
                gt_t = self.dreamsim_proc(gt_imgs)
                pr_t = self.dreamsim_proc(pr_imgs)

                if isinstance(gt_t, torch.Tensor) and gt_t.dim() == 5 and gt_t.shape[1] == 1:
                    gt_t = gt_t.squeeze(1)
                if isinstance(pr_t, torch.Tensor) and pr_t.dim() == 5 and pr_t.shape[1] == 1:
                    pr_t = pr_t.squeeze(1)

                if not (isinstance(gt_t, torch.Tensor) and gt_t.dim() == 4):
                    raise ValueError(f"Unexpected batched gt_t shape/type: {type(gt_t)} {getattr(gt_t, 'shape', None)}")
                if not (isinstance(pr_t, torch.Tensor) and pr_t.dim() == 4):
                    raise ValueError(f"Unexpected batched pr_t shape/type: {type(pr_t)} {getattr(pr_t, 'shape', None)}")

            gt_t = gt_t.to(self.device, dtype=self.dreamsim_dtype)
            pr_t = pr_t.to(self.device, dtype=self.dreamsim_dtype)

            with torch.inference_mode():
                dist = self.dreamsim_model(gt_t, pr_t)

            if isinstance(dist, torch.Tensor):
                d = dist.detach().float().view(-1).cpu().tolist()
            else:
                d = [float(dist)]

            for key, dist_val in zip(batch, d):
                dist_val = float(dist_val)
                out[key].dreamsim_dist = dist_val
                sim = 1.0 - dist_val
                if sim < 0.0:
                    sim = 0.0
                elif sim > 1.0:
                    sim = 1.0
                out[key].dreamsim_sim = float(sim)
