# -*- coding: utf-8 -*-
"""
SignSpeak AI - Translation Evaluation Metrics
=============================================
Calculates BLEU, ROUGE-L, and Exact-Match Accuracy on held-out evaluation pairs.

IMPORTANT EVALUATION NOTICE:
----------------------------
Automatic NLP metrics (BLEU, ROUGE-L, Exact Match) evaluate string-level and n-gram overlap
against written English reference targets. They do NOT by themselves establish authentic or
linguistically complete Indian Sign Language translation, which intrinsically involves
spatial syntax, facial expressions, and non-manual markers that text strings cannot capture.
"""

from typing import Any, Dict, List, Optional
import re

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK_BLEU = True
except ImportError:
    _HAS_NLTK_BLEU = False


def normalize_text_for_eval(text: str) -> str:
    """Standardizes punctuation, casing, and spacing for fair evaluation."""
    if not text:
        return ""
    t = text.strip().lower()
    # Normalize punctuation spacing
    t = re.sub(r"\s+", " ", t)
    return t


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Computes exact-match accuracy between predictions and references.
    Case-insensitive, normalized punctuation.
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    matches = 0
    for pred, ref in zip(predictions, references):
        p_norm = normalize_text_for_eval(pred)
        r_norm = normalize_text_for_eval(ref)
        if p_norm == r_norm:
            matches += 1
    return float(matches / len(predictions))


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Computes corpus-level BLEU score (0.0 to 100.0).
    Prefers sacrebleu if available, falling back to NLTK sentence_bleu with smoothing.
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0

    # Ensure non-empty references
    preds = [p.strip() if p.strip() else "." for p in predictions]
    refs = [[r.strip() if r.strip() else "."] for r in references]

    if _HAS_SACREBLEU:
        try:
            # sacrebleu corpus_bleu expects references as list of lists of references
            # transposed: list of reference streams
            ref_streams = [[r[0] for r in refs]]
            bleu = sacrebleu.corpus_bleu(preds, ref_streams)
            return float(bleu.score)
        except Exception:
            pass

    if _HAS_NLTK_BLEU:
        try:
            smoothie = SmoothingFunction().method1
            scores = []
            for p, r in zip(preds, refs):
                p_tokens = p.lower().split()
                r_tokens = [r[0].lower().split()]
                s = sentence_bleu(r_tokens, p_tokens, smoothing_function=smoothie)
                scores.append(s * 100.0)
            return float(sum(scores) / len(scores))
        except Exception:
            pass

    # Pure Python basic n-gram overlap BLEU fallback if external libraries are unavailable
    return _simple_bleu_fallback(preds, [r[0] for r in refs])


def _simple_bleu_fallback(preds: List[str], refs: List[str]) -> float:
    """Minimal unigram precision fallback."""
    scores = []
    for p, r in zip(preds, refs):
        p_words = normalize_text_for_eval(p).split()
        r_words = set(normalize_text_for_eval(r).split())
        if not p_words:
            scores.append(0.0)
            continue
        overlap = sum(1 for w in p_words if w in r_words)
        scores.append((overlap / len(p_words)) * 100.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """
    Computes average ROUGE-L F1 score (0.0 to 100.0).
    Uses rouge_score if available, otherwise LCS-based fallback.
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0

    if _HAS_ROUGE:
        try:
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            f1_scores = []
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                f1_scores.append(score["rougeL"].fmeasure * 100.0)
            return float(sum(f1_scores) / len(f1_scores))
        except Exception:
            pass

    # LCS Fallback
    return _simple_rouge_l_fallback(predictions, references)


def _simple_rouge_l_fallback(preds: List[str], refs: List[str]) -> float:
    """Pure Python Longest Common Subsequence (LCS) ROUGE-L."""
    f1_scores = []
    for p, r in zip(preds, refs):
        p_tokens = normalize_text_for_eval(p).split()
        r_tokens = normalize_text_for_eval(r).split()
        if not p_tokens or not r_tokens:
            f1_scores.append(0.0)
            continue

        # Compute LCS length
        m, n = len(p_tokens), len(r_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if p_tokens[i] == r_tokens[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        lcs = dp[m][n]
        prec = lcs / m if m > 0 else 0.0
        rec = lcs / n if n > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1 * 100.0)
    return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0


def evaluate_predictions(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Computes all standard translation evaluation metrics:
      - exact_match: exact match accuracy (0.0 to 1.0)
      - bleu: corpus BLEU score (0.0 to 100.0)
      - rouge_l: average ROUGE-L F1 (0.0 to 100.0)
    """
    em = compute_exact_match(predictions, references)
    bleu = compute_bleu(predictions, references)
    rouge_l = compute_rouge_l(predictions, references)

    return {
        "exact_match": round(em, 4),
        "bleu": round(bleu, 2),
        "rouge_l": round(rouge_l, 2),
        "sample_count": len(predictions),
    }


def generate_qualitative_review(
    glosses: List[str],
    predictions: List[str],
    references: List[str]
) -> List[Dict[str, Any]]:
    """
    Pairs input glosses, model predictions, and ground-truth references
    for transparent manual qualitative inspection.
    """
    review = []
    for g, p, r in zip(glosses, predictions, references):
        is_match = normalize_text_for_eval(p) == normalize_text_for_eval(r)
        review.append({
            "gloss": g,
            "prediction": p,
            "reference": r,
            "exact_match": is_match,
        })
    return review
