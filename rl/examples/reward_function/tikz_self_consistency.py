# TikZ Visual Consistency Reward Function (Image2Code)
#
# Self Consistency RL reward function with three main components:
# 1. Compilation reward: success (+compile_success_reward) or failure (compile_fail_penalty)
#    - Format errors are naturally included in compilation failure (can't extract/compile invalid code)
# 2. Visual reward: weighted combination of SIGLIP (semantic) and LPIPS (structural) similarity
#    - Only computed when compilation succeeds
# 3. Code Consistency reward: similarity between Code and Code' (Self Consistency)
#    - Uses TED (Token Edit Distance) and CrystalBLEU (CrystalBLEU weighted higher)
#    - Only computed when visual_consistency > threshold (default 0.7)
#    - Prevents reward hacking by ensuring code structure consistency
#
# NOTE:
# - This file is meant to be used with EasyR1/verl reward_function loader:
#     worker.reward.reward_function=./examples/reward_function/tikz_self_consistency.py:compute_score

from __future__ import annotations

import os
import re
import math
import uuid
import hashlib
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

try:
    from .code_metrics import CodeMetrics
except ImportError:
    import sys
    from pathlib import Path as PathLib
    code_metrics_path = PathLib(__file__).parent / "code_metrics.py"
    if code_metrics_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("code_metrics", code_metrics_path)
        code_metrics_module = importlib.util.module_from_spec(spec)
        sys.modules["code_metrics"] = code_metrics_module
        spec.loader.exec_module(code_metrics_module)
        CodeMetrics = code_metrics_module.CodeMetrics
    else:
        CodeMetrics = None

REWARD_NAME = "tikz_cycle_consistency"
REWARD_TYPE = "batch"

_global_render_service = None
_global_visual_calculator = None
_global_code_metrics = None

_global_lru_cache: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
_GLOBAL_CACHE_MAX = 4096

# -------------------------
# LRU cache helpers
# -------------------------
def _lru_get(key: str) -> Optional[Dict[str, float]]:
    v = _global_lru_cache.get(key)
    if v is not None:
        _global_lru_cache.move_to_end(key)
    return v


def _lru_put(key: str, val: Dict[str, float]) -> None:
    _global_lru_cache[key] = val
    _global_lru_cache.move_to_end(key)
    if len(_global_lru_cache) > _GLOBAL_CACHE_MAX:
        _global_lru_cache.popitem(last=False)


# -------------------------
# Robust extraction / sanitize
# -------------------------
_FENCE_RE = re.compile(r"```(?:latex|tex|tikz)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# -------------------------
# Structure/Format Checking
# -------------------------
def _check_structure_format(code: str, response: str) -> Dict[str, float]:
    """
    Check structure and format compliance.
    Returns dict with format_score (penalty-based, range [-0.3, +0.2]).
    """
    format_score = 0.0
    violations = []
    
    fence_matches = list(_FENCE_RE.finditer(response if response else ""))
    has_opening_fence = bool(re.search(r"```(?:latex|tex|tikz)?", response or "", re.IGNORECASE))
    has_closing_fence = bool(re.search(r"```\s*$", response or "", re.MULTILINE))
    
    if len(fence_matches) == 0:
        if has_opening_fence and not has_closing_fence:
            format_score -= 0.3
            violations.append("unclosed_fence")
        else:
            format_score -= 0.2
            violations.append("no_fence")
    elif len(fence_matches) > 1:
        format_score -= 0.2
        violations.append("multiple_fences")
    elif len(fence_matches) == 1:
        format_score += 0.1
    
    # Clip to [-0.3, +0.2] range
    format_score = max(-0.3, min(0.2, format_score))
    
    return {
        "format_score": float(format_score),
        "violations": violations,
    }


# -------------------------
# Length/Truncation Checking
# -------------------------
def _check_truncation_risk(code: str, response: str) -> float:
    """
    Check if code is likely truncated.
    Returns truncation_risk in [0, 1] (higher = more likely truncated).
    """
    if not code or not response:
        return 1.0
    
    has_closing_fence = bool(re.search(r"```\s*$", response, re.MULTILINE))
    if not has_closing_fence:
        return 0.8
    
    code_lower = code.lower()
    has_document_end = bool(re.search(r"\\end\s*\{document\}", code_lower))
    has_tikzpicture_end = bool(re.search(r"\\end\s*\{tikzpicture\}", code_lower))
    
    if not has_document_end and not has_tikzpicture_end:
        return 0.6
    
    open_braces = code.count("{")
    close_braces = code.count("}")
    if abs(open_braces - close_braces) > 2:
        return 0.4
    
    return 0.0


def _estimate_code_length(code: str, tokenizer=None) -> Dict[str, Any]:
    """
    Estimate code length in tokens and characters.
    Returns dict with token_count, char_count.
    """
    char_count = len(code) if code else 0
    
    # Try to estimate tokens (rough: ~4 chars per token for LaTeX)
    token_count = char_count // 4
    
    # If tokenizer available, use it
    if tokenizer is not None:
        try:
            tokens = tokenizer.encode(code, add_special_tokens=False)
            token_count = len(tokens)
        except Exception:
            pass
    
    return {
        "token_count": int(token_count),
        "char_count": int(char_count),
    }


# -------------------------
# Code Consistency (Self Consistency RL)
# -------------------------
def _get_code_metrics_service(**kwargs):
    """
    Get or initialize CodeMetrics service for computing code similarity.
    Uses TED (Token Edit Distance) and CrystalBLEU.
    
    Note: This is for comparing two model outputs (Code vs Code'), NOT comparing with GT.
    The corpus is only used by CrystalBLEU to filter out trivial n-grams (common code snippets),
    not for comparison. It's recommended to use a corpus of training set codes.
    """
    global _global_code_metrics
    
    if _global_code_metrics is None:
        try:
            from .code_metrics import patch_nltk_bleu_fraction_for_py312
            patch_nltk_bleu_fraction_for_py312()
        except Exception:
            pass  
        
        try:
            # Check if CodeMetrics is available (imported from local module)
            if CodeMetrics is None:
                print("[tikz_cycle_consistency] WARNING: CodeMetrics not available. Code consistency rewards will use fallback similarity.")
                _global_code_metrics = None
                return None
            
            # Get corpus for CrystalBLEU (used to filter trivial n-grams, NOT for comparison)
            # This should be a corpus of training set codes, not GT codes
            corpus_path = kwargs.get("gt_corpus_path", None) or kwargs.get("corpus_path", None)
            corpus = []
            if corpus_path and os.path.exists(corpus_path):
                    # Load corpus from file - support multiple formats:
                    # 1. Parquet: pandas/pyarrow parquet file (for training data)
                    # 2. JSONL: each line is a JSON object with 'code' or 'response' field
                    # 3. One code per line (if codes are single-line)
                    # 4. Multi-line codes separated by empty lines or special markers
                    try:
                        # Check file extension for parquet
                        file_ext = os.path.splitext(corpus_path)[1].lower()
                        
                        if file_ext == '.parquet':
                            # Parquet format - common for training datasets
                            try:
                                import pandas as pd
                                df = pd.read_parquet(corpus_path)
                                
                                # Try common column names for code
                                code_col = None
                                for col_name in ['answer', 'code', 'response', 'text', 'content', 'output', 'generated_code']:
                                    if col_name in df.columns:
                                        code_col = col_name
                                        break
                                
                                if code_col:
                                    corpus = df[code_col].dropna().astype(str).tolist()
                                    print(f"[tikz_cycle_consistency] Loaded {len(corpus)} codes from parquet file (column: {code_col})")
                                else:
                                    print(f"[tikz_cycle_consistency] WARNING: No code column found in parquet. Available columns: {list(df.columns)}")
                                    # Try to use first text-like column
                                    for col in df.columns:
                                        if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
                                            corpus = df[col].dropna().astype(str).tolist()
                                            print(f"[tikz_cycle_consistency] Using first text column: {col}")
                                            break
                            except ImportError:
                                print(f"[tikz_cycle_consistency] WARNING: pandas not available, cannot read parquet file. Install with: pip install pandas pyarrow")
                            except Exception as e:
                                print(f"[tikz_cycle_consistency] WARNING: Failed to read parquet file: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            # Text-based formats (JSONL, plain text)
                            import json
                            with open(corpus_path, "r", encoding="utf-8") as f:
                                first = f.readline().strip()
                                f.seek(0)
                                
                                if first.startswith("{"):
                                    # JSONL format
                                    for line in f:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        try:
                                            obj = json.loads(line)
                                        except json.JSONDecodeError:
                                            continue
                                        code = obj.get("code") or obj.get("response") or obj.get("text") or obj.get("content")
                                        if code:
                                            corpus.append(str(code))
                                else:
                                    # Plain text format - read all content
                                    content = f.read()
                                
                                # Try splitting by double newlines (common for multi-line code blocks)
                                if '\n\n' in content:
                                    parts = content.split('\n\n')
                                    for part in parts:
                                        part = part.strip()
                                        if part and len(part) > 10:  # Filter out very short fragments
                                            # Check if it looks like LaTeX code
                                            if '\\' in part or 'tikz' in part.lower() or 'document' in part.lower():
                                                corpus.append(part)
                                
                                # If no double newlines found, try reading line by line
                                if not corpus:
                                    f.seek(0)
                                    current_code = []
                                    for line in f:
                                        line = line.rstrip('\n\r')
                                        if line.strip():
                                            current_code.append(line)
                                        else:
                                            # Empty line - end of current code block
                                            if current_code:
                                                code_block = '\n'.join(current_code)
                                                if len(code_block) > 10:  # Filter very short fragments
                                                    corpus.append(code_block)
                                                current_code = []
                                    # Add last code block if file doesn't end with newline
                                    if current_code:
                                        code_block = '\n'.join(current_code)
                                        if len(code_block) > 10:
                                            corpus.append(code_block)
                                
                                # Fallback: if still empty, treat each non-empty line as a code
                                if not corpus:
                                    f.seek(0)
                                    for line in f:
                                        line = line.strip()
                                        if line and len(line) > 10:
                                            corpus.append(line)
                    except Exception as e:
                        print(f"[tikz_cycle_consistency] WARNING: Failed to load corpus from {corpus_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        corpus = []
            
            if not corpus:
                print("[tikz_cycle_consistency] INFO: No corpus provided, CrystalBLEU will use empty corpus (no trivial n-gram filtering)")
            else:
                total_chars = sum(len(c) for c in corpus)
                avg_chars = total_chars / len(corpus) if corpus else 0
                multi_line_count = sum(1 for c in corpus if '\n' in c)
                print(f"[tikz_cycle_consistency] Loaded {len(corpus)} codes from corpus:")
                print(f"  - Total characters: {total_chars}")
                print(f"  - Average length: {avg_chars:.1f} chars")
                print(f"  - Multi-line codes: {multi_line_count}/{len(corpus)} ({multi_line_count/len(corpus)*100:.1f}%)")
                if len(corpus) > 0:
                    sample_code = corpus[0]
                    sample_preview = sample_code[:100].replace('\n', '\\n')
                    print(f"  - Sample code preview: {sample_preview}...")
            
            code_metrics_kwargs = {
                "gt_corpus_for_crystalbleu": corpus,
                "crystal_k": int(kwargs.get("crystal_k", 500)),
                "crystal_n": int(kwargs.get("crystal_n", 4)),
                "crystal_use_cache": bool(kwargs.get("crystal_use_cache", True)),
                "crystal_cache_dir": kwargs.get("crystal_cache_dir", None),
                "token_edit_language": str(kwargs.get("token_edit_language", "en")),
                "token_edit_alpha": float(kwargs.get("token_edit_alpha", 2.0)),
                "token_edit_rho": float(kwargs.get("token_edit_rho", 0.3)),
                "token_edit_deletion": float(kwargs.get("token_edit_deletion", 0.2)),
                "token_edit_insertion": float(kwargs.get("token_edit_insertion", 1.0)),
                "token_edit_tau_for_sim": float(kwargs.get("token_edit_tau_for_sim", 0.4)),
                "crystal_mode": str(kwargs.get("crystal_mode", "sentence")),
            }
            
            # Initialize CodeMetrics with error handling
            try:
                _global_code_metrics = CodeMetrics(**code_metrics_kwargs)
                print(f"[tikz_cycle_consistency] CodeMetrics initialized with {len(corpus)} corpus codes (for trivial n-gram filtering)")
            except TypeError as e:
                # If TypeError occurs, it might be due to parameter mismatch
                # Try with minimal parameters
                print(f"[tikz_cycle_consistency] WARNING: CodeMetrics init failed with full params: {e}")
                print(f"[tikz_cycle_consistency] Attempting with minimal parameters...")
                try:
                    # Try with only required parameters
                    minimal_kwargs = {
                        "gt_corpus_for_crystalbleu": corpus,
                    }
                    _global_code_metrics = CodeMetrics(**minimal_kwargs)
                    print(f"[tikz_cycle_consistency] CodeMetrics initialized with minimal parameters")
                except Exception as e2:
                    raise Exception(f"CodeMetrics initialization failed even with minimal params: {e2}. Original error: {e}")
            except AttributeError as e:
                    error_msg = str(e)
                    if "'NoneType' object has no attribute '__dict__'" in error_msg or "'NoneType' object has no attribute" in error_msg:
                        print(f"[tikz_cycle_consistency] WARNING: CodeMetrics initialization failed due to NoneType error: {e}")
                        print(f"[tikz_cycle_consistency] This might be due to missing dependencies (torchmetrics, crystalbleu, pygments, sacremoses)")
                        print(f"[tikz_cycle_consistency] Attempting to check dependencies...")
                        # Check dependencies
                        missing_deps = []
                        try:
                            import torchmetrics
                        except ImportError:
                            missing_deps.append("torchmetrics")
                        try:
                            import crystalbleu
                        except ImportError:
                            missing_deps.append("crystalbleu")
                        try:
                            import pygments
                        except ImportError:
                            missing_deps.append("pygments")
                        try:
                            import sacremoses
                        except ImportError:
                            missing_deps.append("sacremoses")
                        
                        if missing_deps:
                            raise ImportError(f"Missing dependencies: {', '.join(missing_deps)}. Please install them: pip install {' '.join(missing_deps)}")
                        else:
                            raise Exception(f"CodeMetrics initialization failed with AttributeError: {e}. All dependencies seem to be installed.")
                    else:
                        raise
        except Exception as e:
            print(f"[tikz_cycle_consistency] Failed to initialize CodeMetrics: {e}")
            import traceback
            traceback.print_exc()
            _global_code_metrics = None
    
    return _global_code_metrics


def _compute_code_consistency(code: str, code_prime: Optional[str]) -> Dict[str, float]:
    """
    Compute code consistency between Code and Code' using TED and CrystalBLEU.
    
    This compares two model outputs (Code vs Code'), NOT comparing with GT.
    The corpus used by CrystalBLEU is only for filtering trivial n-grams (common code snippets),
    not for comparison.
    
    Args:
        code: First model output (Code, generated from Image)
        code_prime: Second model output (Code', generated from Image')
    
    Returns dict with:
    - ted_sim: Token Edit Distance similarity [0, 1]
    - crystalbleu: CrystalBLEU score [0, 1]
    - overall: Weighted combination (CrystalBLEU weighted higher)
    """
    if not code or not code_prime:
        return {
            "ted_sim": 0.0,
            "crystalbleu": 0.0,
            "overall": 0.0,
        }
    
    # Get code metrics service (should be initialized by _get_services)
    code_metrics = _global_code_metrics
    if code_metrics is None:
        # Fallback: simple char-3gram similarity
        return {
            "ted_sim": 0.0,
            "crystalbleu": 0.0,
            "overall": _compute_text_similarity_fallback(code, code_prime),
        }
    
    try:
        result = code_metrics.compute_one(code, code_prime)
        
        # Weighted combination: CrystalBLEU (0.7), TED (0.3)
        overall = 0.7 * float(result.crystalbleu) + 0.3 * float(result.token_edit_sim)
        
        return {
            "ted_sim": float(result.token_edit_sim),
            "crystalbleu": float(result.crystalbleu),
            "overall": float(max(0.0, min(1.0, overall))),
        }
    except Exception as e:
        error_str = str(e)
        if "_normalize" in error_str or "Fraction" in error_str:
            pass
        else:
            # Only print unexpected errors
            print(f"[tikz_cycle_consistency] Error computing code consistency: {e}")
            import traceback
            traceback.print_exc()
        # Fallback
        return {
            "ted_sim": 0.0,
            "crystalbleu": 0.0,
            "overall": _compute_text_similarity_fallback(code, code_prime),
        }


# -------------------------
# Text Similarity Fallback (for recoverable errors only)
# -------------------------
def _compute_text_similarity_fallback(code: str, reference_code: Optional[str] = None) -> float:
    """
    Compute text-side similarity fallback using char-3gram Jaccard.
    Only used for recoverable errors (missing_closing, syntax_error).
    Returns similarity in [0, 1].
    """
    if not code:
        return 0.0
    
    if not reference_code:
        has_tikz = "tikz" in code.lower()
        has_draw = "\\draw" in code or "\\node" in code
        return 0.3 if (has_tikz and has_draw) else 0.0
    
    def get_ngrams(text: str, n: int = 3) -> set:
        text = text.lower().replace(" ", "").replace("\n", "")
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams_code = get_ngrams(code)
    ngrams_ref = get_ngrams(reference_code)
    
    if not ngrams_code or not ngrams_ref:
        return 0.0
    
    intersection = len(ngrams_code & ngrams_ref)
    union = len(ngrams_code | ngrams_ref)
    
    if union == 0:
        return 0.0
    
    similarity = intersection / union
    return float(max(0.0, min(1.0, similarity)))

def extract_code_from_response(response: str) -> str:
    """
    Extract tikz/latex code from a model response.
    Fixes common issues like:
      - leading ```latex without closing fence
      - stray backticks that cause "Missing \\begin{document}"
      - CRLF and weird invisible chars
    """
    if response is None:
        return ""
    s = str(response)

    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1)

    s = s.strip()
    s = re.sub(r"^\s*`+\s*(latex|tex|tikz)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*```(?:latex|tex|tikz)?\s*\n", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n\s*```\s*$", "\n", s)
    s = s.replace("\r", "\n").replace("\x00", "")

    return s.strip()


def _is_full_latex_doc(code: str) -> bool:
    c = (code or "").lower()
    return ("\\documentclass" in c) or ("\\begin{document}" in c)


def _inject_nuclear_no_page(code: str) -> str:
    r"""
    For non-standalone docs, inject nuclear-level code to eliminate page number/header artifacts.
    
    This matches the logic in clean.py:inject_nuclear_no_page():
    - If standalone: return as-is (standalone already has no page numbers)
    - If non-standalone: inject \makeatletter ... \pagestyle{empty} before \begin{document}
    - Also add \thispagestyle{empty} before \end{document}
    
    This ensures article/book/etc. documents render without page numbers/headers.
    """
    if not code:
        return code
    if "{standalone}" in code:
        return code
    if re.search(r"\\begin\s*\{document\}", code) is None:
        return code

    nuclear_code = (
        r"\makeatletter" + "\n" +
        r"\let\ps@plain\ps@empty" + "\n" +
        r"\let\ps@headings\ps@empty" + "\n" +
        r"\let\ps@firstpage\ps@empty" + "\n" +
        r"\makeatother" + "\n" +
        r"\pagestyle{empty}" + "\n" +
        r"\begin{document}"
    )
    new_code = re.sub(r"\\begin\s*\{document\}", lambda _: nuclear_code, code, count=1)
    new_code = new_code.replace(r"\end{document}", r"\thispagestyle{empty}" + "\n" + r"\end{document}")
    return new_code


# -------------------------
# Preambles / shims
# -------------------------
_MINIMAL_PREAMBLE = r"""
\usepackage{tikz}
\usepackage{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,calc,positioning,fit,backgrounds,intersections,
                shapes.geometric,shapes.misc,patterns,
                decorations.pathmorphing,decorations.pathreplacing,matrix}
"""

_COMPAT_SHIMS = r"""
% ---- standalonepicture compatibility: map to tikzpicture ----
\makeatletter
\@ifundefined{standalonepicture}{
  \newenvironment{standalonepicture}[1][]{
    \begin{tikzpicture}[#1]
  }{
    \end{tikzpicture}
  }
}{}
\makeatother

% ---- common unknown tikz keys: provide fallback styles to avoid fatal pgfkeys ----
\pgfkeysifdefined{/tikz/op amp}{}{%
  \tikzset{op amp/.style={draw, trapezium, trapezium left angle=70, trapezium right angle=110,
                         minimum width=1.2cm, minimum height=0.8cm}}
}

\pgfkeysifdefined{/tikz/grayblob}{}{%
  \tikzset{grayblob/.style={fill=gray, draw=none, opacity=0.35}}
}

% ---- pgfplots key compat: mark options sometimes triggers key errors ----
\pgfkeysifdefined{/pgfplots/mark options}{}{%
  \pgfkeys{/pgfplots/mark options/.code={}}
}
"""

def _need_compat_retry(log_text: str) -> bool:
    if not log_text:
        return False
    pats = [
        "Missing \\begin{document}",
        "I do not know the key '/tikz/op amp'",
        "Environment standalonepicture undefined",
        "I do not know the key '/pgfplots/mark options'",
        "I do not know the key '/tikz/grayblob'",
        "``latex",
        "```latex",
    ]
    return any(p in log_text for p in pats)


# -------------------------
# Apptainer detection
# -------------------------
def _pick_apptainer_bin(apptainer_bin: Optional[str] = None) -> Optional[str]:
    if apptainer_bin:
        p = Path(apptainer_bin)
        if p.exists() and p.is_file():
            return str(p)

    for envk in ("APPTAINER_BIN", "SINGULARITY_BIN"):
        v = os.environ.get(envk, "").strip()
        if v:
            p = Path(v)
            if p.exists() and p.is_file():
                return str(p)

    p = shutil.which("apptainer") or shutil.which("singularity")
    if p:
        return p

    for c in (
        "/usr/bin/apptainer",
        "/usr/local/bin/apptainer",
        "/opt/apptainer/bin/apptainer",
        "/usr/bin/singularity",
        "/usr/local/bin/singularity",
    ):
        if os.path.exists(c):
            return c

    return None


# -------------------------
# Rendering service
# -------------------------
class TikZRenderService:
    """
    Render pipeline:
      code -> main.tex -> (pdf) -> (png)

    If use_apptainer=True:
      apptainer exec <sif> pdflatex ...
      apptainer exec <sif> convert ...
    else:
      run pdflatex/convert directly in current environment.
      (Recommended when training already runs inside SIF.)
    """
    def __init__(
        self,
        sif_path: str,
        base_tmp_dir: str = "/tmp/tikz_render",
        timeout_sec: int = 30,
        convert_timeout_sec: int = 30,
        density: int = 200,
        quality: int = 95,
        border: int = 2,
        enable_nuclear_no_page: bool = True,
        apptainer_bin: Optional[str] = None,
        extra_binds: Optional[List[str]] = None,
        common_binds: Optional[List[str]] = None,
        debug_env: bool = False,
        use_cleanenv: bool = True,
        use_apptainer: bool = True,
        latex_engine: str = "pdflatex",
        lualatex_fallback: bool = False,
    ):
        self.sif_path = str(sif_path)
        self.base_tmp_dir = str(Path(base_tmp_dir) / str(os.getpid()))
        Path(self.base_tmp_dir).mkdir(parents=True, exist_ok=True)

        self.timeout_sec = int(timeout_sec)
        self.convert_timeout_sec = int(convert_timeout_sec)
        self.density = int(density)
        self.quality = int(quality)
        self.border = int(border)
        self.enable_nuclear_no_page = bool(enable_nuclear_no_page)

        self.apptainer_bin = apptainer_bin
        self.extra_binds = extra_binds or []
        self.common_binds = common_binds or ["/tmp:/tmp"]
        self.debug_env = bool(debug_env)
        self.use_cleanenv = bool(use_cleanenv)
        self.use_apptainer = bool(use_apptainer)

        self.latex_engine = str(latex_engine or "pdflatex")
        self.lualatex_fallback = bool(lualatex_fallback)

    def _build_binds(self, job_dir: Path, texmf_var: Path) -> List[str]:
        binds: List[str] = []
        binds.append(f"{job_dir}:{job_dir}")
        binds.append(f"{texmf_var}:{texmf_var}")

        # Common binds
        for b in self.common_binds:
            binds.append(b)

        binds.extend(self.extra_binds)
        return binds

    def _run_exec(
        self,
        binds: List[str],
        env_kv: Dict[str, str],
        argv: List[str],
        cwd: str,
        timeout: int,
    ) -> subprocess.CompletedProcess:
        # Ensure isolated LaTeX caches etc.
        run_env = os.environ.copy()
        run_env.update(env_kv)

        if not self.use_apptainer:
            # Direct execution (FAST) - assumes pdflatex/convert exist in current environment
            if self.debug_env:
                print(f"[tikz_self_consistency] [exec-direct] argv={' '.join(argv)} cwd={cwd} env_kv={env_kv}")
            return subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                encoding="latin-1",
                text=True,
                timeout=timeout,
                env=run_env,
            )

        # Apptainer execution
        apptainer_cmd = _pick_apptainer_bin(self.apptainer_bin)
        if apptainer_cmd is None:
            raise RuntimeError(
                "Cannot find apptainer/singularity. Please pass apptainer_bin=/usr/bin/apptainer or ensure it is in PATH."
            )

        cmd = [apptainer_cmd, "exec"]
        if self.use_cleanenv:
            cmd.append("--cleanenv")

        for k, v in env_kv.items():
            cmd += ["--env", f"{k}={v}"]

        for b in binds:
            cmd += ["--bind", b]

        cmd += [self.sif_path]
        cmd += argv

        if self.debug_env:
            print(f"[tikz_self_consistency] [exec-apptainer] cmd={' '.join(cmd[:18])} ... (len={len(cmd)})")
            print(f"[tikz_self_consistency] [exec-apptainer] cwd={cwd}")
            print(f"[tikz_self_consistency] [exec-apptainer] env_kv={env_kv}")

        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            encoding="latin-1",
            text=True,
            timeout=timeout,
            env=run_env,
        )

    def _write_wrapped_tex(self, tex_path: Path, code_body: str, compat: bool = False) -> str:
        """
        If code_body is not a full LaTeX doc, wrap it with standalone + preamble.
        compat=True injects compatibility shims.
        Returns the final tex string written.
        """
        if _is_full_latex_doc(code_body):
            tex_code = code_body
        else:
            pre = _MINIMAL_PREAMBLE
            if compat:
                pre = _MINIMAL_PREAMBLE + "\n" + _COMPAT_SHIMS
            tex_code = (
                "\\documentclass[tikz]{standalone}\n"
                f"{pre}\n"
                "\\begin{document}\n"
                f"{code_body}\n"
                "\\end{document}\n"
            )
        if self.enable_nuclear_no_page:
            tex_code = _inject_nuclear_no_page(tex_code)

        tex_path.write_text(tex_code, encoding="utf-8")
        return tex_code

    def render(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        if code is None:
            return False, None, "empty_code"
        code = str(code).strip()
        if not code:
            return False, None, "empty_code"

        job_dir = Path(self.base_tmp_dir) / uuid.uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=True)

        tex_basename = "main"
        tex_path = job_dir / f"{tex_basename}.tex"
        pdf_path = job_dir / f"{tex_basename}.pdf"
        png_path = job_dir / f"{tex_basename}.png"

        try:
            # Write tex (minimal wrapper)
            tex_code = self._write_wrapped_tex(tex_path, code, compat=False)

            texmf_var = job_dir / "texmf_var"
            texmf_var.mkdir(parents=True, exist_ok=True)

            binds = self._build_binds(job_dir, texmf_var)
            env_kv = {
                "TEXMFVAR": str(texmf_var),
                "TEXMFHOME": str(texmf_var),
            }

            # 1) compile (pdflatex by default)
            compile_argv = [
                self.latex_engine,
                "--shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"{tex_basename}.tex",
            ]
            r = self._run_exec(binds=binds, env_kv=env_kv, argv=compile_argv, cwd=str(job_dir), timeout=self.timeout_sec)

            if r.returncode != 0 or (not pdf_path.exists()):
                # Decide compat retry for non-model-root causes
                log_tail = (r.stdout or "")[-5000:] + "\n" + (r.stderr or "")[-2000:]
                did_retry = False
                if (not _is_full_latex_doc(code)) and _need_compat_retry(log_tail):
                    did_retry = True
                    tex_code = self._write_wrapped_tex(tex_path, code, compat=True)
                    r2 = self._run_exec(binds=binds, env_kv=env_kv, argv=compile_argv, cwd=str(job_dir), timeout=self.timeout_sec)
                    if r2.returncode == 0 and pdf_path.exists():
                        r = r2
                    else:
                        r = r2  # keep last logs

                # Optional: lualatex fallback (off by default)
                if (not pdf_path.exists()) and self.lualatex_fallback:
                    lualatex_argv = [
                        "lualatex",
                        "--shell-escape",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        f"{tex_basename}.tex",
                    ]
                    r3 = self._run_exec(binds=binds, env_kv=env_kv, argv=lualatex_argv, cwd=str(job_dir), timeout=self.timeout_sec)
                    if r3.returncode == 0 and pdf_path.exists():
                        r = r3

                if r.returncode != 0 or (not pdf_path.exists()):
                    log_tail = (r.stdout or "")[-5000:] + "\n" + (r.stderr or "")[-2000:]
                    print(f"[tikz_self_consistency] LaTeX compilation failed. returncode={r.returncode}, pdf_exists={pdf_path.exists()}")
                    if r.stderr:
                        print(f"[tikz_self_consistency] LaTeX stderr (last 800 chars): {r.stderr[-800:]}")
                    if r.stdout:
                        print(f"[tikz_self_consistency] LaTeX stdout (last 800 chars): {r.stdout[-800:]}")
                    self._cleanup_dir(job_dir)
                    return False, None, "latex_error"

            # 2) convert to png
            # Logic matches clean.py:run_compilation():
            # - If standalone: direct conversion (no trim, no border)
            # - If non-standalone (article/book/etc.): trim white edges + add 2px white border
            is_standalone = "{standalone}" in tex_code
            if is_standalone:
                # Standalone: keep as-is, no trim/border needed
                convert_argv = [
                    "convert",
                    "-density", str(self.density),
                    str(pdf_path) + "[0]",
                    "-background", "white", "-flatten",
                    "-quality", str(self.quality),
                    str(png_path),
                ]
            else:
                # Non-standalone: trim white edges and add uniform white border
                # Order matches clean.py: -trim +repage, then -bordercolor white -border 2
                convert_argv = [
                    "convert",
                    "-density", str(self.density),
                    str(pdf_path) + "[0]",
                    "-background", "white", "-flatten",
                    "-trim", "+repage",
                    "-bordercolor", "white",
                    "-border", str(self.border),
                    "-quality", str(self.quality),
                    str(png_path),
                ]

            if not pdf_path.exists():
                print(f"[tikz_self_consistency] Convert skipped: PDF does not exist (LaTeX compilation failed)")
                self._cleanup_dir(job_dir)
                return False, None, "pdf_not_exists"
            
            r_convert = self._run_exec(binds=binds, env_kv=env_kv, argv=convert_argv, cwd=str(job_dir), timeout=self.convert_timeout_sec)
            if not png_path.exists():
                print(f"[tikz_self_consistency] Convert failed. returncode={r_convert.returncode}")
                if r_convert.stderr:
                    print(f"[tikz_self_consistency] Convert stderr (last 800 chars): {r_convert.stderr[-800:]}")
                if r_convert.stdout:
                    print(f"[tikz_self_consistency] Convert stdout (last 800 chars): {r_convert.stdout[-800:]}")
                self._cleanup_dir(job_dir)
                return False, None, "convert_error"
            
            try:
                png_size = png_path.stat().st_size
                if png_size < 1024:
                    print(f"[tikz_self_consistency] Convert produced suspiciously small PNG: {png_size} bytes (likely error file)")
                    self._cleanup_dir(job_dir)
                    return False, None, "png_too_small"
                
                from PIL import Image
                try:
                    with Image.open(png_path) as im:
                        im.verify()
                except Exception as e:
                    print(f"[tikz_self_consistency] Convert produced invalid PNG (PIL verify failed): {e}")
                    self._cleanup_dir(job_dir)
                    return False, None, "png_invalid"
            except Exception as e:
                print(f"[tikz_self_consistency] Failed to validate PNG: {e}")
                self._cleanup_dir(job_dir)
                return False, None, "png_validation_error"

            return True, str(png_path), None

        except subprocess.TimeoutExpired:
            print(f"[tikz_self_consistency] Render timeout after {self.timeout_sec}s")
            self._cleanup_dir(job_dir)
            return False, None, "timeout"
        except Exception as e:
            print(f"[tikz_self_consistency] Exception during render: {type(e).__name__}: {e}")
            import traceback
            print(f"[tikz_self_consistency] Traceback: {traceback.format_exc()}")
            self._cleanup_dir(job_dir)
            return False, None, "exception"

    def cleanup_rendered(self, png_path: str) -> None:
        try:
            p = Path(png_path)
            self._cleanup_dir(p.parent)
        except Exception:
            pass

    @staticmethod
    def _cleanup_dir(d: Path) -> None:
        try:
            if not d.exists():
                return
            for child in d.glob("**/*"):
                try:
                    if child.is_file():
                        child.unlink()
                except Exception:
                    pass
            for sub in sorted(d.glob("**/*"), reverse=True):
                try:
                    if sub.is_dir():
                        sub.rmdir()
                except Exception:
                    pass
            try:
                d.rmdir()
            except Exception:
                pass
        except Exception:
            pass


# -------------------------
# Visual similarity
# -------------------------
class VisualSimilarityCalculator:
    def __init__(
        self,
        siglip_model_path: str,
        lpips_net: str = "alex",
        device: str = "cuda",
        torch_home: str = os.environ.get("TORCH_HOME", "/tmp/.torch_cache"),
        input_size: int = 384,
        lpips_tau: float = 0.5,
    ):
        self.siglip_model_path = siglip_model_path
        self.lpips_net = lpips_net
        self.torch_home = torch_home
        self.input_size = int(input_size)
        self.lpips_tau = float(lpips_tau)

        self.device = "cpu"
        try:
            import torch
            if device == "cuda" and torch.cuda.is_available():
                self.device = "cuda"
        except Exception:
            self.device = "cpu"

        os.environ.setdefault("TORCH_HOME", self.torch_home)
        Path(self.torch_home).mkdir(parents=True, exist_ok=True)

        self.siglip_processor = None
        self.siglip_model = None
        self.lpips_model = None
        self._tfm = None

        self._load_siglip()
        self._load_lpips()

    def _load_siglip(self) -> None:
        try:
            from transformers import SiglipModel, SiglipImageProcessor
            self.siglip_processor = SiglipImageProcessor.from_pretrained(self.siglip_model_path)
            self.siglip_model = SiglipModel.from_pretrained(self.siglip_model_path).to(self.device).eval()
            print(f"[tikz_self_consistency] SigLIP loaded: {self.siglip_model_path} on {self.device}")
        except Exception as e:
            print(f"[tikz_self_consistency] SigLIP load failed: {e}")
            self.siglip_model = None
            self.siglip_processor = None

    def _load_lpips(self) -> None:
        try:
            import lpips
            from torchvision import transforms
            self.lpips_model = lpips.LPIPS(net=self.lpips_net).to(self.device).eval()
            self._tfm = transforms.Compose(
                [
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),
                ]
            )
            print(f"[tikz_self_consistency] LPIPS loaded: net={self.lpips_net} size={self.input_size} on {self.device}")
        except Exception as e:
            print(f"[tikz_self_consistency] LPIPS load failed: {e}")
            self.lpips_model = None
            self._tfm = None
    def _rescale_hold(self, s: float, hold: float = 0.8) -> float:
        """
        Rescale similarity with a hold threshold:
        s' = max(0, (s - hold) / (1 - hold))
        Then clamp to [0, 1].
        """
        try:
            s = float(s)
            hold = float(hold)
            if hold >= 1.0:
                return 0.0
            v = (s - hold) / (1.0 - hold)
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            return float(v)
        except Exception:
            return 0.0

    def compute_similarity(
        self,
        image1_path: str,
        image2_path: str,
        semantic_weight: float = 0.4,
        structural_weight: float = 0.6,
        siglip_hold: float = 0.8, 
    ) -> Dict[str, float]:
        semantic = 0.0
        structural = 0.0

        if self.siglip_model is not None:
            semantic = self._siglip_sim01(image1_path, image2_path)
            semantic = self._rescale_hold(semantic, hold=float(siglip_hold))

        if self.lpips_model is not None:
            structural = self._lpips_sim(image1_path, image2_path, tau=self.lpips_tau)

        semantic = float(max(0.0, min(1.0, semantic)))
        structural = float(max(0.0, min(1.0, structural)))

        sw = float(semantic_weight)
        lw = float(structural_weight)
        s = sw + lw
        if s <= 0:
            sw, lw = 0.4, 0.6
            s = 1.0
        sw /= s
        lw /= s

        overall = sw * semantic + lw * structural
        return {
            "semantic": float(semantic),
            "structural": float(structural),
            "overall": float(overall),
        }

    def _siglip_sim01(self, img1_path: str, img2_path: str) -> float:
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            inputs = self.siglip_processor(images=[img1, img2], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                feats = self.siglip_model.get_image_features(**inputs)
                feats = F.normalize(feats, p=2, dim=-1)
                cos = (feats[0] * feats[1]).sum().item()

            return float((float(cos) + 1.0) / 2.0)
        except Exception:
            return 0.0

    def _lpips_sim(self, img1_path: str, img2_path: str, tau: float = 0.5) -> float:
        try:
            import torch
            from PIL import Image

            if self._tfm is None or self.lpips_model is None:
                return 0.0

            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            t1 = self._tfm(img1).unsqueeze(0).to(self.device)
            t2 = self._tfm(img2).unsqueeze(0).to(self.device)
            t1 = t1 * 2 - 1
            t2 = t2 * 2 - 1

            with torch.no_grad():
                d = float(self.lpips_model(t1, t2).item())

            return float(math.exp(-d / float(tau)))
        except Exception:
            return 0.0


# -------------------------
# Global services
# -------------------------
def _get_services(**kwargs):
    global _global_render_service, _global_visual_calculator

    # Initialize code metrics service (lazy loading)
    _get_code_metrics_service(**kwargs)

    if _global_render_service is None:
        _global_render_service = TikZRenderService(
            sif_path=kwargs.get("sif_path", os.environ.get("SIF_PATH", "/path/to/apptainer.sif")),
            base_tmp_dir=kwargs.get("temp_dir", "/tmp/tikz_render"),
            timeout_sec=int(kwargs.get("timeout_sec", 30)),
            convert_timeout_sec=int(kwargs.get("convert_timeout_sec", 30)),
            density=int(kwargs.get("density", 200)),
            quality=int(kwargs.get("quality", 95)),
            border=int(kwargs.get("border", 2)),
            enable_nuclear_no_page=bool(kwargs.get("enable_nuclear_no_page", True)),
            apptainer_bin=kwargs.get("apptainer_bin", None),
            extra_binds=kwargs.get("extra_binds", None),
            common_binds=kwargs.get("common_binds", None),
            debug_env=bool(kwargs.get("debug_env", False)),
            use_cleanenv=bool(kwargs.get("use_cleanenv", True)),
            use_apptainer=bool(kwargs.get("use_apptainer", False)),  # Default False for in-SIF training
            latex_engine=str(kwargs.get("latex_engine", "pdflatex")),
            lualatex_fallback=bool(kwargs.get("lualatex_fallback", False)),
        )

    if _global_visual_calculator is None:
        _global_visual_calculator = VisualSimilarityCalculator(
            siglip_model_path=kwargs.get(
                "siglip_model_path",
                os.environ.get("SIGLIP_MODEL_PATH", "/path/to/siglip-model"),
            ),
            lpips_net=kwargs.get("lpips_net", "alex"),
            device=kwargs.get("device", "cuda"),
            torch_home=kwargs.get(
                "torch_home",
                os.environ.get("TORCH_HOME", "/tmp/.torch_cache"),
            ),
            input_size=int(kwargs.get("size", 384)),
            lpips_tau=float(kwargs.get("tau", 0.5)),
        )

    return _global_render_service, _global_visual_calculator


# -------------------------
# Reward API
# -------------------------
def compute_score(
    reward_inputs: List[Dict[str, Any]],
    siglip_model_path: str = os.environ.get("SIGLIP_MODEL_PATH", "/path/to/siglip-model"),
    lpips_net: str = "alex",
    device: str = "cuda",
    torch_home: str = os.environ.get("TORCH_HOME", "/tmp/.torch_cache"),
    sif_path: str = os.environ.get("SIF_PATH", "/path/to/apptainer.sif"),
    temp_dir: str = "/tmp/tikz_render",
    timeout_sec: int = 20,
    convert_timeout_sec: int = 20,
    density: int = 200,
    quality: int = 95,
    border: int = 2,
    enable_nuclear_no_page: bool = True,
    apptainer_bin: Optional[str] = None,
    extra_binds: Optional[List[str]] = None,
    common_binds: Optional[List[str]] = None,
    image_base_dirs: Optional[List[str]] = None,
    debug_env: bool = False,
    use_cleanenv: bool = True,
    use_apptainer: bool = False,  # IMPORTANT: default to False for in-SIF training
    latex_engine: str = "pdflatex",
    lualatex_fallback: bool = False,
    semantic_weight: float = 0.4,
    structural_weight: float = 0.6,
    compile_success_reward: float = 0.1,
    siglip_hold: float = 0.8, 
    compile_fail_penalty: float = -0.8,
    visual_weight: float = 0.8,
    visual_consistency_threshold: float = 0.5,  # Threshold for computing Code Consistency
    code_weight: float = 0.15,  # Weight for code consistency reward
    code_consistency_threshold: float = 0.6,  # Threshold for code similarity (if allowing penalty)
    code_penalty_max: float = 0.1,  # Maximum penalty for low code consistency
    gt_corpus_path: Optional[str] = None,  # Path to corpus for CrystalBLEU (used to filter trivial n-grams, NOT for comparison)
    corpus_path: Optional[str] = None,  # Alias for gt_corpus_path (recommended: training set codes)
    crystal_k: int = 500,
    crystal_n: int = 4,
    token_edit_tau_for_sim: float = 0.4,
    size: int = 384,
    tau: float = 0.5,
    **kwargs,
) -> List[Dict[str, float]]:

    render_service, visual_calculator = _get_services(
        siglip_model_path=siglip_model_path,
        lpips_net=lpips_net,
        device=device,
        torch_home=torch_home,
        size=size,
        tau=tau,
        sif_path=sif_path,
        temp_dir=temp_dir,
        timeout_sec=timeout_sec,
        convert_timeout_sec=convert_timeout_sec,
        density=density,
        quality=quality,
        border=border,
        enable_nuclear_no_page=enable_nuclear_no_page,
        apptainer_bin=apptainer_bin,
        extra_binds=extra_binds,
        common_binds=common_binds,
        image_base_dirs=image_base_dirs,
        debug_env=debug_env,
        use_cleanenv=use_cleanenv,
        use_apptainer=use_apptainer,
        latex_engine=latex_engine,
        lualatex_fallback=lualatex_fallback,
        gt_corpus_path=gt_corpus_path or corpus_path,
        corpus_path=corpus_path or gt_corpus_path,
        crystal_k=crystal_k,
        crystal_n=crystal_n,
        token_edit_tau_for_sim=token_edit_tau_for_sim,
    )

    outputs: List[Dict[str, float]] = []

    for idx, ri in enumerate(reward_inputs):
        resp = ri.get("response", "")
        image_path = ri.get("image_path", None) or ri.get("images", None)
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0] if len(image_path) > 0 else None

        if not image_path:
            out = {
                "score": float(compile_fail_penalty),
                "overall": float(compile_fail_penalty),
                "r_compile": float(compile_fail_penalty),
                "r_visual": 0.0,
                "r_code": 0.0,
                "code_similarity": 0.0,
                "code_ted_sim": 0.0,
                "code_crystalbleu": 0.0,
                "visual_consistency": 0.0,
                "semantic": 0.0,
                "structural": 0.0,
                "compile_success": 0.0,
            }
            outputs.append(out)
            continue

        if not os.path.exists(image_path):
            if not os.path.isabs(image_path):
                # Try to find image in common data directories
                base_dirs = image_base_dirs if image_base_dirs else [".", "./data", "./data/images", "./images"]
                for base_dir in base_dirs:
                    full_path = os.path.join(base_dir, image_path)
                    if os.path.exists(full_path):
                        image_path = full_path
                        break

            if not os.path.exists(image_path):
                out = {
                    "score": float(compile_fail_penalty),
                    "overall": float(compile_fail_penalty),
                    "r_compile": float(compile_fail_penalty),
                    "r_visual": 0.0,
                    "r_code": 0.0,
                    "code_similarity": 0.0,
                    "code_ted_sim": 0.0,
                    "code_crystalbleu": 0.0,
                    "visual_consistency": 0.0,
                    "semantic": 0.0,
                    "structural": 0.0,
                    "compile_success": 0.0,
                }
                outputs.append(out)
                continue

        code = extract_code_from_response(resp)
        if not code.strip():
            out = {
                "score": float(compile_fail_penalty),
                "overall": float(compile_fail_penalty),
                "r_compile": float(compile_fail_penalty),
                "r_visual": 0.0,
                "r_code": 0.0,
                "code_similarity": 0.0,
                "code_ted_sim": 0.0,
                "code_crystalbleu": 0.0,
                "visual_consistency": 0.0,
                "semantic": 0.0,
                "structural": 0.0,
                "compile_success": 0.0,
            }
            outputs.append(out)
            continue

        code_prime = None
        code_prime_resp = ri.get("code_prime", None) or ri.get("response_prime", None)
        if code_prime_resp:
            code_prime = extract_code_from_response(str(code_prime_resp))
            if not hasattr(compute_score, '_cycle_log_counter'):
                compute_score._cycle_log_counter = 0
            compute_score._cycle_log_counter += 1
            if compute_score._cycle_log_counter % 50 == 0:
                print(f"[Cycle Consistency RL] Sample {compute_score._cycle_log_counter}: code_prime available (len={len(code_prime) if code_prime else 0})")

        cfg_str = (
            f"{size}|{tau}|{siglip_hold}|"
            f"{semantic_weight}|{structural_weight}|{visual_weight}|"
            f"{visual_consistency_threshold}|{code_weight}|"
            f"{compile_success_reward}|{compile_fail_penalty}|"
            f"{density}|{siglip_model_path}|{lpips_net}"
        )

        cfg_hash = hashlib.sha1(cfg_str.encode("utf-8", errors="ignore")).hexdigest()[:8]
        code_prime_hash = ""
        if code_prime:
            code_prime_hash = "|" + hashlib.sha1(code_prime.encode("utf-8", errors="ignore")).hexdigest()[:8]
        cache_key = hashlib.sha1(code.encode("utf-8", errors="ignore")).hexdigest() + "|" + str(image_path) + "|" + cfg_hash + code_prime_hash
        cached = _lru_get(cache_key)
        if cached is not None:
            outputs.append(dict(cached))
            continue

        ok, rendered_png, err_tag = render_service.render(code)
        
        if (not ok) or (rendered_png is None):
            out = {
                "score": float(compile_fail_penalty),
                "overall": float(compile_fail_penalty),
                "r_compile": float(compile_fail_penalty),
                "r_visual": 0.0,
                "r_code": 0.0,

                "cycle_render_ok": 0.0,
                "cycle_has_codeprime": 1.0 if code_prime else 0.0,
                "cycle_vis": 0.0,
                "cycle_vis_gate_pass": 0.0,
                "cycle_aspect_ok": 0.0,
                "cycle_enter": 0.0,
                "cycle_code_len": float(len(code)) if code else 0.0,
                "cycle_codeprime_len": float(len(code_prime)) if code_prime else 0.0,

                "code_similarity": 0.0,
                "code_ted_sim": 0.0,
                "code_crystalbleu": 0.0,
                "visual_consistency": 0.0,
                "semantic": 0.0,
                "structural": 0.0,
                "compile_success": 0.0,
            }
            
            _lru_put(cache_key, out)
            outputs.append(out)
            continue

        r_compile = compile_success_reward
        vis = visual_calculator.compute_similarity(
            image_path,
            rendered_png,
            semantic_weight=float(semantic_weight),
            structural_weight=float(structural_weight),
            siglip_hold=float(siglip_hold),
        )

        r_visual = float(vis["overall"]) * visual_weight
        visual_consistency = float(vis["overall"])
        
        r_code = 0.0
        code_sim_overall = 0.0
        code_ted_sim = 0.0
        code_crystalbleu = 0.0
        
        # Check rendered image aspect ratio BEFORE cleanup
        # If aspect ratio > 15:1, skip code similarity (extreme aspect ratios indicate problematic code)
        aspect_ratio_ok = True
        try:
            from PIL import Image
            img = Image.open(rendered_png)
            w, h = img.size
            if w > 0 and h > 0:
                ar = max(w / h, h / w)
                if ar > 15.0:
                    aspect_ratio_ok = False
                    if not hasattr(compute_score, '_aspect_ratio_skip_counter'):
                        compute_score._aspect_ratio_skip_counter = 0
                    compute_score._aspect_ratio_skip_counter += 1
                    if compute_score._aspect_ratio_skip_counter <= 10:
                        print(f"[Cycle Consistency RL] Skipping code consistency due to extreme aspect ratio: {w}x{h} (ratio={ar:.2f} > 15:1)")
        except Exception:
            pass
        
        render_service.cleanup_rendered(rendered_png)
        
        has_codeprime = 1.0 if code_prime else 0.0
        vis_gate_pass = 1.0 if (visual_consistency > float(visual_consistency_threshold)) else 0.0
        aspect_ok = 1.0 if aspect_ratio_ok else 0.0
        cycle_enter = 1.0 if (vis_gate_pass > 0 and has_codeprime > 0 and aspect_ok > 0) else 0.0

        if visual_consistency > float(visual_consistency_threshold) and code_prime and aspect_ratio_ok:
            if not hasattr(compute_score, '_cycle_code_consistency_counter'):
                compute_score._cycle_code_consistency_counter = 0
            compute_score._cycle_code_consistency_counter += 1
            if compute_score._cycle_code_consistency_counter % 20 == 0:
                print(f"[Cycle Consistency RL] Computing code similarity (sample {compute_score._cycle_code_consistency_counter}): "
                      f"visual_consistency={visual_consistency:.3f} > threshold={visual_consistency_threshold}, "
                      f"code_len={len(code)}, code_prime_len={len(code_prime) if code_prime else 0}")
            
            code_consistency = _compute_code_consistency(code, code_prime)
            code_sim_overall = code_consistency["overall"]
            code_ted_sim = code_consistency["ted_sim"]
            code_crystalbleu = code_consistency["crystalbleu"]
            
            if code_sim_overall >= float(code_consistency_threshold):
                reward_ratio = (code_sim_overall - code_consistency_threshold) / (1.0 - code_consistency_threshold)
                r_code = reward_ratio * code_weight
            else:
                penalty_ratio = (code_consistency_threshold - code_sim_overall) / code_consistency_threshold
                r_code = -penalty_ratio * code_penalty_max
                r_code = max(r_code, -code_penalty_max)
            
            if compute_score._cycle_code_consistency_counter % 20 == 0:
                print(f"[Cycle Consistency RL] Code similarity results: "
                      f"overall={code_sim_overall:.3f}, TED={code_ted_sim:.3f}, CrystalBLEU={code_crystalbleu:.3f}, "
                      f"r_code={r_code:.4f}")
        elif code_prime:
            if aspect_ratio_ok:
                if not hasattr(compute_score, '_cycle_skip_counter'):
                    compute_score._cycle_skip_counter = 0
                compute_score._cycle_skip_counter += 1
                if compute_score._cycle_skip_counter % 50 == 0:
                    print(f"[Cycle Consistency RL] Skipping code consistency (sample {compute_score._cycle_skip_counter}): "
                          f"visual_consistency={visual_consistency:.3f} <= threshold={visual_consistency_threshold}")
        
        total = r_compile + r_visual + r_code
        
        # Clip to [-1.0, 1.0]
        total = max(-1.0, min(1.0, total))

        out = {
            "score": float(total),
            "overall": float(total),
            "r_compile": float(r_compile),
            "r_visual": float(r_visual),
            "r_code": float(r_code),

            "cycle_render_ok": 1.0,
            "cycle_has_codeprime": float(has_codeprime),
            "cycle_vis": float(visual_consistency),
            "cycle_vis_gate_pass": float(vis_gate_pass),
            "cycle_aspect_ok": float(aspect_ok),
            "cycle_enter": float(cycle_enter),
            "cycle_code_len": float(len(code)) if code else 0.0,
            "cycle_codeprime_len": float(len(code_prime)) if code_prime else 0.0,

            "code_similarity": float(code_sim_overall),
            "code_ted_sim": float(code_ted_sim),
            "code_crystalbleu": float(code_crystalbleu),
            "visual_consistency": float(visual_consistency),
            "semantic": float(vis["semantic"]),
            "structural": float(vis["structural"]),
            "compile_success": 1.0,
        }

        _lru_put(cache_key, out)
        outputs.append(out)

    return outputs
