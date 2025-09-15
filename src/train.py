from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, cast, Protocol, Mapping, Any
import os
import importlib

import torch
from torch.optim.adamw import AdamW
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from tqdm import tqdm
from torch.amp.autocast_mode import autocast as amp_autocast
from torch.amp.grad_scaler import GradScaler as AmpGradScaler
from contextlib import nullcontext

from .eval.text_utils import ngram_repeat_rate
from .eval.metrics import squad_em_f1, rougeL_f1, sst2_label_from_text


@dataclass
class TrainConfig:
    task: str
    dataset_size: str
    model_size: str
    model_id: str
    seq_len: int
    train_epochs: int
    lr: float
    micro_batch: int
    accum_steps: int
    eval_trials: int
    device: str
    backend: str
    seed: int
    nan_guard: bool
    # Data subsampling
    train_fraction: float
    eval_fraction: float
    max_train_samples: int
    max_eval_samples: int
    # I/O
    save_dir: str
    eval_only: bool
    ckpt_path: str
    # Logging
    log_every: int
    # Boids regularization
    boids_on: bool
    boids_weight: float
    boids_align: float
    boids_sep: float
    boids_entropy: float


MODEL_SIZES: Dict[str, Dict[str, int]] = {
    "tiny": {"hidden_dim": 384, "ffn_dim": 1536, "num_layers": 4},
    "small": {"hidden_dim": 768, "ffn_dim": 3072, "num_layers": 6},
    "base": {"hidden_dim": 1024, "ffn_dim": 4096, "num_layers": 8},
}


# HF save/load helpers
class HFSeq2SeqLike(Protocol):
    def __call__(self, *, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., labels: Optional[torch.Tensor] = ..., **kwargs: Any) -> Any: ...
    def parameters(self) -> Any: ...
    def train(self, mode: bool = ...) -> Any: ...
    def eval(self) -> Any: ...
    def generate(self, *, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., max_new_tokens: Optional[int] = ..., do_sample: Optional[bool] = ..., **kwargs: Any) -> torch.Tensor: ...
    def save_pretrained(self, save_directory: str) -> Any: ...


def hf_load(model_id_or_dir: str, device: torch.device) -> Tuple[Any, HFSeq2SeqLike]:
    tok_mod = importlib.import_module("transformers.models.auto.tokenization_auto")
    AutoTokenizer = getattr(tok_mod, "AutoTokenizer")
    tok: Any = AutoTokenizer.from_pretrained(model_id_or_dir)
    if getattr(tok, "pad_token", None) is None:
        pad_fallback = getattr(tok, "eos_token", None) or getattr(tok, "sep_token", None) or getattr(tok, "unk_token", None)
        if pad_fallback is not None:
            setattr(tok, "pad_token", pad_fallback)
    mdl_mod = importlib.import_module("transformers.models.auto.modeling_auto")
    AutoModelForSeq2SeqLM = getattr(mdl_mod, "AutoModelForSeq2SeqLM")
    mdl_any: Any = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_dir)
    # Force router logits in config if supported
    try:
        setattr(mdl_any.config, "output_router_logits", True)
    except Exception:
        pass
    mdl = cast(HFSeq2SeqLike, mdl_any.to(device))
    return tok, mdl


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="squad", choices=["squad", "cnndm", "sst2"])  # 形式のみ
    p.add_argument("--dataset_size", type=str, default="small", choices=["small", "full"])
    p.add_argument("--model_size", type=str, default="small", choices=list(MODEL_SIZES.keys()))
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--train_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=8)
    p.add_argument("--eval_trials", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", type=str, default="hf_moe", choices=["hf_moe"], help="Model backend (HF Switch only)")
    p.add_argument("--model_id", type=str, default="google/switch-base-16", help="HF model id or local dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nan_guard", type=lambda x: x.lower() == "true", default=True, help="Guard against NaN/Inf loss and grads by skipping the step")
    # Data subsampling options
    p.add_argument("--train_fraction", type=float, default=1.0, help="Use this fraction of the training set (0-1]")
    p.add_argument("--eval_fraction", type=float, default=1.0, help="Use this fraction of the eval set (0-1]")
    p.add_argument("--max_train_samples", type=int, default=0, help="Cap the number of training samples (0=disabled)")
    p.add_argument("--max_eval_samples", type=int, default=0, help="Cap the number of eval samples (0=disabled)")
    # I/O options
    p.add_argument("--save_dir", type=str, default="", help="Directory to save final checkpoint (if non-empty)")
    p.add_argument("--eval_only", type=lambda x: x.lower() == "true", default=False, help="Load checkpoint and run evaluation only")
    p.add_argument("--ckpt_path", type=str, default="", help="Path to a HF directory to load (required for --eval_only)")
    # Logging
    p.add_argument("--log_every", type=int, default=0, help="Print a one-line progress log every N steps (0=disabled)")
    # Boids regularization options
    p.add_argument("--boids_on", type=lambda x: x.lower() == "true", default=True, help="Boids regularization is mandatory; must be true")
    p.add_argument("--boids_weight", type=float, default=0.01)
    p.add_argument("--boids_align", type=float, default=1.0)
    p.add_argument("--boids_sep", type=float, default=1.0)
    p.add_argument("--boids_entropy", type=float, default=0.0)
    a = p.parse_args()
    # Enforce Boids mandatory
    if not bool(getattr(a, "boids_on", True)):
        raise SystemExit("Boids regularization is mandatory for this project. Remove --boids_on false and use a Switch-Transformer model (e.g., google/switch-base-16).")
    return TrainConfig(**vars(a))


def main():
    cfg = parse_args()
    device = torch.device(cfg.device)
    # Reproducibility (best-effort)
    # Use getattr to appease strict type checkers on torch stubs
    getattr(torch, "manual_seed")(int(cfg.seed))
    if torch.cuda.is_available():
        getattr(torch.cuda, "manual_seed_all")(int(cfg.seed))
    # HF model/tokenizer
    if cfg.eval_only:
        if not cfg.ckpt_path:
            raise SystemExit("--eval_only requires --ckpt_path to be set to a HF directory")
        tok, model = hf_load(cfg.ckpt_path, device)
    else:
        tok, model = hf_load(cfg.model_id, device)

    # Data loaders (HF tokenizer id to align special tokens)
    from .data import hf_loader as _hf_loader
    train_loader, dev_loader, tok = _hf_loader.build_hf_dataloaders(
        cfg.task,
        cfg.dataset_size,
        cfg.seq_len,
        cfg.micro_batch,
        0,
        cfg.train_fraction,
        cfg.eval_fraction,
        cfg.max_train_samples,
        cfg.max_eval_samples,
        model_id=cfg.model_id,
    )

    # Router logits capture via forward hooks (fallback if outputs lack router info)
    router_captures: List[torch.Tensor] = []
    def _router_collect(x: Any, out_list: List[torch.Tensor]) -> None:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            out_list.append(x)
        elif isinstance(x, (list, tuple)):
            for xi in cast(List[Any] | Tuple[Any, ...], x):
                _router_collect(xi, out_list)
        elif isinstance(x, dict):
            for v in cast(Dict[Any, Any], x).values():
                _router_collect(v, out_list)

    def _make_router_hook():
        def _hook(module: Any, inputs: Tuple[Any, ...], output: Any) -> None:
            tmp: List[torch.Tensor] = []
            _router_collect(output, tmp)
            for t in tmp:
                if t.dim() >= 3 and int(t.shape[-1]) >= 2 and torch.is_floating_point(t):
                    router_captures.append(t.detach())
        return _hook

    hook_handles: List[Any] = []
    try:
        hook = _make_router_hook()
        for name, m in cast(Any, model).named_modules():
            lname = str(name).lower()
            cls = m.__class__.__name__.lower() if hasattr(m, "__class__") else ""
            if ("router" in lname) or ("router" in cls) or ("switch" in lname) or ("switch" in cls) or ("moe" in lname) or ("moe" in cls) or ("gate" in lname) or ("gate" in cls) or ("gating" in lname) or ("gating" in cls) or ("expert" in lname) or ("expert" in cls):
                try:
                    handle = m.register_forward_hook(hook)
                    hook_handles.append(handle)
                except Exception:
                    pass
    except Exception:
        pass

    # 評価専用モード: 先にチェックポイントをロード
    goto_eval = bool(cfg.eval_only)

    opt = AdamW(model.parameters(), lr=cfg.lr) if not goto_eval else AdamW(model.parameters(), lr=cfg.lr)
    # Loss is returned by HF model when passing labels
    model.train()
    global_step = 0
    # total_steps not used explicitly
    scaler = AmpGradScaler(enabled=torch.cuda.is_available())

    if not goto_eval:
        for epoch in range(cfg.train_epochs):
            pbar = tqdm(train_loader, desc=f"epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            for it, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                pad_id_any: Any = getattr(tok, "pad_token_id", None)
                pad_id = int(pad_id_any) if pad_id_any is not None else -100
                attn_mask = (input_ids != pad_id).to(dtype=torch.long)
                # Mixed precision forward
                amp_ctx = amp_autocast(device_type="cuda", enabled=torch.cuda.is_available()) if torch.cuda.is_available() else nullcontext()
                with amp_ctx:
                    # Clear previous captures
                    if router_captures:
                        router_captures.clear()
                    # Always request router logits (Boids mandatory)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        labels=labels,
                        output_router_logits=True,
                        return_dict=True,
                    )
                    outputs_any: Any = outputs
                    loss: torch.Tensor = cast(torch.Tensor, outputs_any.loss)

                    # Collect router probs per layer robustly
                    def _collect_tensors(x: Any, out_list: List[torch.Tensor]) -> None:
                        import torch as _torch
                        if isinstance(x, _torch.Tensor):
                            out_list.append(x)
                        elif isinstance(x, (list, tuple)):
                            for xi in cast(List[Any] | Tuple[Any, ...], x):
                                _collect_tensors(xi, out_list)
                        elif isinstance(x, dict):
                            for v in cast(Dict[Any, Any], x).values():
                                _collect_tensors(v, out_list)

                    router_probs: List[torch.Tensor] = []
                    # Common attribute names across HF MoE variants
                    cand_names = [
                        "router_logits",
                        "encoder_router_logits",
                        "decoder_router_logits",
                        "router_probabilities",
                        "encoder_router_probabilities",
                        "decoder_router_probabilities",
                    ]
                    collected_raw: List[torch.Tensor] = []
                    for nm in cand_names:
                        val = getattr(outputs_any, nm, None)
                        if val is not None:
                            _collect_tensors(val, collected_raw)
                    # Fallback: some models may store in outputs.__dict__
                    if not collected_raw and hasattr(outputs_any, "__dict__"):
                        for k, v in getattr(outputs_any, "__dict__").items():
                            if "router" in str(k):
                                _collect_tensors(v, collected_raw)

                    # Convert collected candidates to (B,T,E) probabilities
                    B = int(attn_mask.size(0))
                    T = int(attn_mask.size(1))
                    def _standardize_to_bte(x: torch.Tensor) -> List[torch.Tensor]:
                        out: List[torch.Tensor] = []
                        if not torch.is_floating_point(x):
                            return out
                        dims = list(x.shape)
                        if len(dims) < 3:
                            return out
                        # Find batch and seq axes matching attn_mask
                        try:
                            b_axes = [i for i, s in enumerate(dims) if int(s) == B]
                            t_axes = [i for i, s in enumerate(dims) if int(s) == T]
                        except Exception:
                            return out
                        if not b_axes or not t_axes:
                            return out
                        for bi in b_axes:
                            for ti in t_axes:
                                if bi == ti:
                                    continue
                                # Expert axis: choose any remaining axis with size >= 2
                                e_axes = [i for i in range(len(dims)) if i not in (bi, ti) and int(dims[i]) >= 2]
                                if not e_axes:
                                    continue
                                ei = e_axes[0]
                                # Move to (B,T,E)
                                perm = [bi, ti, ei] + [k for k in range(len(dims)) if k not in (bi, ti, ei)]
                                xt = x.permute(*perm)
                                # Keep only first three dims as (B,T,E) by flattening the rest into E
                                if xt.dim() > 3:
                                    new_b = int(xt.shape[0])
                                    new_t = int(xt.shape[1])
                                    new_e = int(torch.tensor(list(xt.shape[2:])).prod().item()) if xt.shape[2:] else int(xt.shape[2])
                                    xt = xt.reshape(new_b, new_t, new_e)
                                out.append(torch.softmax(xt, dim=-1))
                                return out
                        return out

                    for t in collected_raw:
                        for std in _standardize_to_bte(t):
                            if int(std.size(0)) == B and int(std.size(1)) == T and int(std.size(2)) >= 2:
                                router_probs.append(std)

                    # Fallback to captured tensors from router modules
                    if not router_probs and router_captures:
                        for t in router_captures:
                            for std in _standardize_to_bte(t):
                                if int(std.size(0)) == B and int(std.size(1)) == T and int(std.size(2)) >= 2:
                                    router_probs.append(std)

                    # Filter to tensors with last-dim >= 2 (expert dimension)
                    router_probs = [t for t in router_probs if t.dim() >= 3 and int(t.shape[-1]) >= 2]

                    if not router_probs:
                        try:
                            keys = list(outputs_any.keys()) if hasattr(outputs_any, "keys") else []
                            print(f"[boids] router tensors not found. outputs keys={keys}")
                        except Exception:
                            pass
                        raise RuntimeError(
                            "Router logits were not returned by the model. Boids is mandatory. Use a Switch-Transformer model (e.g., google/switch-base-16) and ensure output_router_logits is supported."
                        )

                    from .regularizers.boids import boids_regularize
                    reg, boids_comps = boids_regularize(
                        router_probs,
                        attn_mask,
                        weight=cfg.boids_weight,
                        w_align=cfg.boids_align,
                        w_separation=cfg.boids_sep,
                        w_entropy=cfg.boids_entropy,
                    )
                    loss = loss + reg

                # NaN/Inf guard on loss (pre-backward)
                if cfg.nan_guard and not bool(torch.isfinite(loss)):
                    # Skip this batch safely
                    opt.zero_grad(set_to_none=True)
                    # Minimal, impersonal notice
                    print(f"[nan-guard] non-finite loss at step {global_step}, batch {it}: skip")
                    scaler.update()
                    continue

                # mypy/pyright: scale() return type lacks backward signature -> cast via a tiny protocol
                class _BackpropLike(Protocol):
                    def backward(self) -> None: ...
                scaled = cast(_BackpropLike, scaler.scale(loss / cfg.accum_steps))
                scaled.backward()
                if (it + 1) % cfg.accum_steps == 0:
                    scaler.unscale_(opt)
                    # Optional: check grads before stepping
                    if cfg.nan_guard:
                        any_bad = False
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            g = p.grad
                            if not bool(torch.isfinite(g).all()):
                                any_bad = True
                                break
                        if any_bad:
                            opt.zero_grad(set_to_none=True)
                            scaler.update()
                            print(f"[nan-guard] non-finite grad at step {global_step}: skip step")
                            continue
                    clip_grad_norm(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                global_step += 1
                postfix: Dict[str, object] = {"loss": float(loss.item())}
                # 追加: Boidsロス内訳をコンパクトに表示（無害なオーバーヘッド）
                try:
                    align_v = float(boids_comps.get('align', torch.tensor(0.0)).item() if hasattr(boids_comps.get('align', 0.0), 'item') else boids_comps.get('align', 0.0))
                    sep_v = float(boids_comps.get('separation', torch.tensor(0.0)).item() if hasattr(boids_comps.get('separation', 0.0), 'item') else boids_comps.get('separation', 0.0))
                    ent_v = float(boids_comps.get('entropy', torch.tensor(0.0)).item() if hasattr(boids_comps.get('entropy', 0.0), 'item') else boids_comps.get('entropy', 0.0))
                    reg_w = float(reg.item()) if hasattr(reg, 'item') else float(reg)
                    postfix.update({"boids_align": round(align_v, 4), "boids_sep": round(sep_v, 4), "boids_ent": round(ent_v, 4), "boids_reg": round(reg_w, 4)})
                except Exception:
                    pass
                class _TqdmLike(Protocol):
                    def set_postfix(self, ordered_dict: Mapping[str, object] | None = ..., refresh: Optional[bool] = True, **kwargs: object) -> None: ...
                cast(_TqdmLike, pbar).set_postfix(postfix)
                # Optional plain-text progress for non-interactive consoles
                if cfg.log_every > 0 and (it % cfg.log_every == 0):
                    tot = len(train_loader)
                    try:
                        align_v = float(boids_comps.get('align', torch.tensor(0.0)).item() if hasattr(boids_comps.get('align', 0.0), 'item') else boids_comps.get('align', 0.0))
                        sep_v = float(boids_comps.get('separation', torch.tensor(0.0)).item() if hasattr(boids_comps.get('separation', 0.0), 'item') else boids_comps.get('separation', 0.0))
                        ent_v = float(boids_comps.get('entropy', torch.tensor(0.0)).item() if hasattr(boids_comps.get('entropy', 0.0), 'item') else boids_comps.get('entropy', 0.0))
                        reg_w = float(reg.item()) if hasattr(reg, 'item') else float(reg)
                        print(f"[train] epoch={epoch} step={it+1}/{tot} loss={float(loss.item()):.4f} boids_reg={reg_w:.4f} (align={align_v:.4f}, sep={sep_v:.4f}, ent={ent_v:.4f})")
                    except Exception:
                        print(f"[train] epoch={epoch} step={it+1}/{tot} loss={float(loss.item()):.4f}")

    # 学習後にチェックポイント保存（任意）
    if (not goto_eval) and cfg.save_dir:
        try:
            os.makedirs(cfg.save_dir, exist_ok=True)
            model.save_pretrained(cfg.save_dir)
            getattr(tok, "save_pretrained")(cfg.save_dir)
            print(f"[save] HF checkpoint dir: {cfg.save_dir}")
        except Exception as e:
            print(f"[save] failed: {e}")

    # Quick eval: generate on a small dev slice and compute task metrics
    model.eval()
    rep2: float = 0.0
    rep3: float = 0.0
    em_list: List[float] = []
    f1_list: List[float] = []
    rl_list: List[float] = []
    acc_list: List[float] = []
    with torch.no_grad():
        for _ in range(max(1, cfg.eval_trials)):
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                pad_id_any: Any = getattr(tok, "pad_token_id", None)
                pad_id = int(pad_id_any) if pad_id_any is not None else -100
                attn_mask = (input_ids != pad_id).to(dtype=torch.long)
                gen = model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=64, do_sample=False)
                # repetition metric on the first sample tokens
                pad_pred_ids: List[int] = [int(x) for x in gen[0]]
                rep2 = ngram_repeat_rate(pad_pred_ids, 2)
                rep3 = ngram_repeat_rate(pad_pred_ids, 3)
                # per-sample metrics using decoded strings
                for i in range(int(input_ids.size(0))):
                    # decode target
                    lbl = batch["labels"][i]
                    tgt_ids: List[int] = [int(x) for x in lbl[lbl != -100]]
                    tgt_txt = getattr(tok, "decode")(tgt_ids, skip_special_tokens=True)
                    prd_txt = getattr(tok, "decode")(gen[i], skip_special_tokens=True)
                    if cfg.task == "squad":
                        em, f1 = squad_em_f1(prd_txt, tgt_txt)
                        em_list.append(em); f1_list.append(f1)
                    elif cfg.task == "cnndm":
                        rl = rougeL_f1(prd_txt, tgt_txt)
                        rl_list.append(rl)
                    elif cfg.task == "sst2":
                        gt = sst2_label_from_text(tgt_txt)
                        pd = sst2_label_from_text(prd_txt)
                        if gt is not None and pd is not None:
                            acc_list.append(1.0 if gt == pd else 0.0)
                break  # one batch is enough per trial
            break

    summary: Dict[str, float] = {"repeat_2gram": rep2, "repeat_3gram": rep3}
    if cfg.task == "squad":
        summary.update({
            "squad_em": float(sum(em_list) / max(1, len(em_list))),
            "squad_f1": float(sum(f1_list) / max(1, len(f1_list))),
        })
    elif cfg.task == "cnndm":
        summary.update({"rougeL_f1": float(sum(rl_list) / max(1, len(rl_list)))})
    elif cfg.task == "sst2":
        summary.update({"acc": float(sum(acc_list) / max(1, len(acc_list)))})
    print(summary)


if __name__ == "__main__":
    main()
