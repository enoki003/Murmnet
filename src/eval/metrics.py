from typing import List, Tuple, Optional, Dict, Mapping, Protocol

import torch
import re
from rouge_score import rouge_scorer  # type: ignore[reportMissingImports]


class _RougeScore(Protocol):
    precision: float
    recall: float
    fmeasure: float


class _RougeScorerLike(Protocol):
    def score(self, target: str, prediction: str) -> Mapping[str, _RougeScore]: ...


def self_consistency_rate(logits_list: List[torch.Tensor]) -> float:
    # logits_list: List[Tensor(B, T, V)]
    if len(logits_list) < 2:
        return 1.0
    preds = [l.argmax(dim=-1) for l in logits_list]
    base = preds[0]
    agree = [(p == base).float().mean().item() for p in preds[1:]]
    return float(sum(agree) / len(agree))


def embedding_variance(hidden_states: torch.Tensor) -> float:
    # hidden_states: (B, T, H)
    var = hidden_states.float().var(dim=(0, 1)).mean().item()
    return float(var)


def expert_entropy(util_probs: torch.Tensor) -> float:
    # util_probs: (E,)
    p = util_probs + 1e-9
    ent = -(p * (p.log())).sum().item()
    return float(ent)


# --- Task metrics (lightweight) ---
_WS = re.compile(r"\s+")

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _WS.sub(" ", s).strip()
    return s

def squad_em_f1(pred: str, gold: str) -> Tuple[float, float]:
    p = _normalize_text(pred)
    g = _normalize_text(gold)
    em = 1.0 if p == g else 0.0
    p_tokens = p.split()
    g_tokens = g.split()
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return em, 0.0
    common: Dict[str, int] = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    precision = match / max(1, len(p_tokens))
    recall = match / max(1, len(g_tokens))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return float(em), float(f1)

_rouge_scorer: _RougeScorerLike = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # type: ignore[reportUnknownMemberType]

def rougeL_f1(pred: str, ref: str) -> float:
    score = _rouge_scorer.score(ref, pred)["rougeL"]
    return float(score.fmeasure)

def sst2_label_from_text(text: str) -> Optional[int]:
    t = text.lower()
    if "positive" in t:
        return 1
    if "negative" in t:
        return 0
    return None
