#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
from functools import cached_property
from hashlib import md5
from itertools import chain, tee
from pathlib import Path
from pickle import dump, load
import math
import os
import re


# ---------- deps: CrystalBLEU ----------
try:
    from crystalbleu import corpus_bleu
except Exception:
    corpus_bleu = None

try:
    # HF cache helper (optional)
    from huggingface_hub import cached_assets_path
except Exception:
    cached_assets_path = None

try:
    from pygments.lexers.markup import TexLexer
    from pygments.token import Comment, Name, Text
except Exception:
    TexLexer = None
    Comment = None
    Name = None
    Text = None

try:
    from sacremoses import MosesTokenizer
except Exception:
    MosesTokenizer = None

# ---------- deps: Token Edit Distance (EED over TeX tokens) ----------
try:
    from torchmetrics.text import ExtendedEditDistance
    from torchmetrics.functional.text.eed import (
        _compute_sentence_statistics,
        _preprocess_en,
        _preprocess_ja,
    )
    from torchmetrics.functional.text.helper import _validate_inputs
except Exception:
    ExtendedEditDistance = None


# =========================
# Result dataclass
# =========================
@dataclass
class CodeMetricResult:
    token_edit_dist: float = float("nan")         # token_edit_dist_norm * ref_len, smaller=better
    token_edit_dist_norm: float = float("nan")    # raw EED output, smaller=better (0 means identical)
    token_edit_sim: float = 0.0                   # mapped to (0,1], higher=better

    # Backward-compatible aliases (if you previously used ted_*)
    ted_dist: float = float("nan")
    ted_sim: float = 0.0

    # CrystalBLEU
    crystalbleu: float = 0.0                      # 0..1 higher=better
    crystalbleu_mode: str = "sentence"            # sentence/corpus


# =========================
# Common utils
# =========================
def extract_document_body(tex: str) -> str:
    m = re.search(
        r"\\begin\{document\}(.*)\\end\{document\}",
        tex,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else tex


def strip_latex_comments(tex: str) -> str:
    """
    Remove LaTeX comments. Keep escaped percent: \%
    NOTE: This is line-based; verbatim edge cases are not handled (usually fine for TikZ).
    """
    out_lines = []
    for line in tex.splitlines():
        i = 0
        cut = None
        while i < len(line):
            if line[i] == "%":
                # ignore escaped \%
                if i > 0 and line[i - 1] == "\\":
                    i += 1
                    continue
                cut = i
                break
            i += 1
        if cut is not None:
            line = line[:cut]
        out_lines.append(line)
    return "\n".join(out_lines)


def normalize_tex(tex: str) -> str:
    tex = tex.replace("\r\n", "\n").replace("\r", "\n")
    tex = strip_latex_comments(tex)
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return tex.strip()


# =========================
# CrystalBLEU helpers (adopted from nltk style)
# =========================
def pad_sequence(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, **kwargs):
    sequence = pad_sequence(sequence, n, **kwargs)
    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


# =========================
# CrystalBLEU
# =========================
class CrystalBLEU:
    """
    CrystalBLEU wrapper adapted for LaTeX/TikZ:
      - TexLexer tokenize (robust comment filtering)
      - Text-like tokens are Moses-tokenized
      - Trivially shared n-grams are computed in a streaming way (no huge all_ngrams list)
      - Cache key is computed without sorting the full corpus (scalable)
    """

    def __init__(
        self,
        corpus: List[str],
        k: int = 500,
        n: int = 4,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
    ):
        if corpus_bleu is None:
            raise ImportError("crystalbleu is not installed: pip install crystalbleu")
        if TexLexer is None or MosesTokenizer is None:
            raise ImportError("pygments and sacremoses are required: pip install pygments sacremoses")

        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()
        self.use_cache = bool(use_cache)
        self.corpus = list(corpus)
        self.k = int(k)
        self.n = int(n)

        self._cache_dir = None
        if cached_assets_path is not None:
            try:
                self._cache_dir = Path(cached_assets_path(library_name="evaluate", namespace="crystalbleu_latex"))
            except Exception:
                self._cache_dir = None
        if self._cache_dir is None:
            if cache_dir is not None:
                self._cache_dir = Path(cache_dir)
            else:
                self._cache_dir = Path(os.path.expanduser("~/.cache/crystalbleu_latex"))

        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _is_comment_token(self, tokentype) -> bool:
        return (Comment is not None) and (tokentype in Comment)

    def _is_text_like_token(self, tokentype) -> bool:
        if Text is not None and tokentype in Text:
            return True
        if Name is not None:
            if tokentype in Name.Attribute or tokentype in Name.Builtin:
                return True
        return False

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        norm = normalize_tex(text)

        for tokentype, value in self.lexer.get_tokens(norm):
            if not value or not value.strip():
                continue
            if self._is_comment_token(tokentype):
                continue

            v = value.strip()
            if not v:
                continue

            if self._is_text_like_token(tokentype):
                tokens.extend(self.tokenizer.tokenize(v))
            else:
                tokens.append(v)

        return tokens

    def _corpus_fingerprint(self) -> str:
        h = md5()
        h.update(f"k={self.k};n={self.n};len={len(self.corpus)}".encode("utf-8"))
        for s in self.corpus:
            ss = normalize_tex(s)
            b = ss.encode("utf-8", errors="ignore")
            h.update(len(b).to_bytes(8, "little", signed=False))
            h.update(b[:4096])
            h.update(md5(b).digest())
        return h.hexdigest()

    @cached_property
    def trivially_shared_ngrams(self) -> Dict[Tuple[str, ...], int]:
        cache_file = self._cache_dir / f"trivial_{self._corpus_fingerprint()}.pkl"

        if self.use_cache and cache_file.is_file():
            with open(cache_file, "rb") as f:
                return load(f)

        freq: Counter = Counter()
        for tex in self.corpus:
            toks = self._tokenize(tex)
            if not toks:
                continue
            for o in range(1, self.n + 1):
                freq.update(ngrams(toks, o))

        trivial = dict(freq.most_common(self.k))

        if self.use_cache:
            with open(cache_file, "wb") as f:
                dump(trivial, f)

        return trivial

    def score_sentence(self, ref: str, hyp: str) -> float:
        ref_toks = self._tokenize(ref)
        hyp_toks = self._tokenize(hyp)
        if not ref_toks or not hyp_toks:
            return 0.0
        return float(
            corpus_bleu(
                list_of_references=[[ref_toks]],
                hypotheses=[hyp_toks],
                ignoring=self.trivially_shared_ngrams,
            )
        )

    def score_corpus(self, refs: List[str], hyps: List[str]) -> float:
        assert len(refs) == len(hyps)
        list_of_references = [[self._tokenize(r)] for r in refs]
        hypotheses = [self._tokenize(h) for h in hyps]
        return float(
            corpus_bleu(
                list_of_references=list_of_references,
                hypotheses=hypotheses,
                ignoring=self.trivially_shared_ngrams,
            )
        )


# =========================
# Token Edit Distance (EED on TeX tokens)
# =========================
class TokenEditDistance(ExtendedEditDistance):
    def __init__(self, *args, **kwargs):
        if ExtendedEditDistance is None:
            raise ImportError("torchmetrics is required: pip install torchmetrics")
        if TexLexer is None:
            raise ImportError("pygments is required: pip install pygments")
        super().__init__(*args, **kwargs)
        self.lexer = TexLexer()

    @staticmethod
    def _is_comment(tokentype) -> bool:
        return (Comment is not None) and (tokentype in Comment)

    @staticmethod
    def _is_text(tokentype) -> bool:
        return (Text is not None) and (tokentype in Text)

    def tokenize_to_tokens(self, text: str, language: str) -> List[str]:
        norm = normalize_tex(text)
        tokens: List[str] = []

        for tokentype, value in self.lexer.get_tokens(norm):
            if not value or not value.strip():
                continue
            if self._is_comment(tokentype):
                continue

            v = value.strip()
            if not v:
                continue

            if self._is_text(tokentype):
                if language == "en":
                    preprocess_function = _preprocess_en
                elif language == "ja":
                    preprocess_function = _preprocess_ja
                else:
                    raise ValueError(f"language must be en/ja, got {language}")
                tokens.extend(preprocess_function(v).split())
            else:
                tokens.extend(v.split())

        return tokens

    def _preprocess_sentences(self, preds, target, language):
        target, preds = _validate_inputs(hypothesis_corpus=preds, ref_corpus=target)

        def to_eed_string(text: str) -> str:
            toks = self.tokenize_to_tokens(text, language=language)
            return " " + " ".join(toks) + " "

        preds = [to_eed_string(pred) for pred in preds]
        target = [[to_eed_string(ref) for ref in reference] for reference in target]
        return preds, target

    def update(self, preds, target):
        preds, target = self._preprocess_sentences(preds, target, self.language)
        if self.sentence_eed is None:
            self.sentence_eed = []
        if 0 in (len(preds), len(target[0])):
            return self.sentence_eed
        for hypothesis, target_words in zip(preds, target):
            score = _compute_sentence_statistics(
                hypothesis, target_words, self.alpha, self.rho, self.deletion, self.insertion
            )
            self.sentence_eed.append(score)
        return self.sentence_eed

    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs).item()


def eed_dist_to_sim(eed_dist_norm: float, tau: float = 0.3) -> float:
    if not math.isfinite(eed_dist_norm):
        return 0.0
    if tau <= 0:
        return 0.0
    d = max(0.0, float(eed_dist_norm))
    return float(math.exp(-d / tau))


# =========================
# Top-level wrapper
# =========================
class CodeMetrics:
    def __init__(
        self,
        gt_corpus_for_crystalbleu: List[str],
        crystal_k: int = 500,
        crystal_n: int = 4,
        crystal_use_cache: bool = True,
        crystal_cache_dir: Optional[str] = None,
        token_edit_language: str = "en",
        token_edit_alpha: float = 2.0,
        token_edit_rho: float = 0.3,
        token_edit_deletion: float = 0.2,
        token_edit_insertion: float = 1.0,
        token_edit_tau_for_sim: float = 0.4,
        crystal_mode: str = "sentence",
    ):
        self.cb = CrystalBLEU(
            corpus=gt_corpus_for_crystalbleu,
            k=crystal_k,
            n=crystal_n,
            use_cache=crystal_use_cache,
            cache_dir=crystal_cache_dir,
        )
        self.crystal_mode = crystal_mode

        self.token_edit_tau_for_sim = float(token_edit_tau_for_sim)

        self.token_edit_metric = TokenEditDistance(
            language=token_edit_language,
            alpha=token_edit_alpha,
            rho=token_edit_rho,
            deletion=token_edit_deletion,
            insertion=token_edit_insertion,
        )

    def compute_one(self, gt_tex: str, pred_tex: str) -> CodeMetricResult:
        gt_body = normalize_tex(extract_document_body(gt_tex))
        pr_body = normalize_tex(extract_document_body(pred_tex))

        self.token_edit_metric.reset()

        eed_dist_norm = float(self.token_edit_metric(preds=[pr_body], target=[[gt_body]]))
        if not math.isfinite(eed_dist_norm):
            eed_dist_norm = float("nan")

        try:
            ref_tokens = self.token_edit_metric.tokenize_to_tokens(gt_body, language=self.token_edit_metric.language)
            ref_len = max(len(ref_tokens), 1)
        except Exception:
            ref_len = 1

        token_edit_dist = float(eed_dist_norm) * float(ref_len) if math.isfinite(eed_dist_norm) else float("nan")
        token_edit_sim = eed_dist_to_sim(eed_dist_norm, tau=self.token_edit_tau_for_sim)

        cb = float(self.cb.score_sentence(gt_body, pr_body))

        return CodeMetricResult(
            token_edit_dist=token_edit_dist,
            token_edit_dist_norm=float(eed_dist_norm),
            token_edit_sim=float(token_edit_sim),
            ted_dist=token_edit_dist,
            ted_sim=float(token_edit_sim),
            crystalbleu=cb,
            crystalbleu_mode="sentence",
        )

    def compute_corpus_crystalbleu(self, gts: List[str], preds: List[str]) -> float:
        gts2 = [normalize_tex(extract_document_body(x)) for x in gts]
        preds2 = [normalize_tex(extract_document_body(x)) for x in preds]
        return float(self.cb.score_corpus(gts2, preds2))
