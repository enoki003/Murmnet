import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, cast

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler  # type: ignore
from contextlib import nullcontext
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .models.moe import MoEConfig, TinyMoETransformer
from .regularizers.boids import BoidsConfig, BoidsRegularizer
from .eval.metrics import self_consistency_rate, expert_entropy
from .eval.moe_utils import overuse_rate
from .eval.text_utils import ngram_repeat_rate
from .loss.boids_router_loss import BoidsRouterLoss


@dataclass
class TrainConfig:
    task: str
    dataset_size: str
    model_size: str
    num_experts: int
    top_k: int
    router_dropout: float
    load_balance_coef: float
    seq_len: int
    train_epochs: int
    lr: float
    micro_batch: int
    accum_steps: int
    boids_on: bool
    boids_lambda_c: float
    boids_lambda_s: float
    boids_lambda_a: float
    boids_k_nn: int
    boids_sep_tau: float
    boids_warmup_frac: float
    eval_trials: int
    device: str
    backend: str


MODEL_SIZES = {
    "tiny": dict(hidden_dim=384, ffn_dim=1536, num_layers=4),
    "small": dict(hidden_dim=768, ffn_dim=3072, num_layers=6),
    "base": dict(hidden_dim=1024, ffn_dim=4096, num_layers=8),
}


def make_model(cfg: TrainConfig, vocab_size: Optional[int] = None) -> TinyMoETransformer:
    base = MODEL_SIZES[cfg.model_size]
    mcfg = MoEConfig(
        model_size=cfg.model_size,
        hidden_dim=base["hidden_dim"],
        ffn_dim=base["ffn_dim"],
        num_layers=base["num_layers"],
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        router_dropout=cfg.router_dropout,
    load_balance_coef=cfg.load_balance_coef,
    vocab_size=vocab_size or 32000,
    )
    return TinyMoETransformer(mcfg)


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="squad", choices=["squad", "cnndm", "sst2"])  # 形式のみ
    p.add_argument("--dataset_size", type=str, default="small", choices=["small", "full"])
    p.add_argument("--model_size", type=str, default="small", choices=list(MODEL_SIZES.keys()))
    p.add_argument("--num_experts", type=int, default=16)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--router_dropout", type=float, default=0.1)
    p.add_argument("--load_balance_coef", type=float, default=0.01)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--train_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--micro_batch", type=int, default=2)
    p.add_argument("--accum_steps", type=int, default=8)
    p.add_argument("--boids_on", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--boids_lambda_c", type=float, default=0.1)
    p.add_argument("--boids_lambda_s", type=float, default=0.05)
    p.add_argument("--boids_lambda_a", type=float, default=0.01)
    p.add_argument("--boids_k_nn", type=int, default=8)
    p.add_argument("--boids_sep_tau", type=float, default=1.5)
    p.add_argument("--boids_warmup_frac", type=float, default=0.1)
    p.add_argument("--eval_trials", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", type=str, default="tiny", choices=["tiny", "hf_moe"], help="Model backend: tiny (this repo) or hf_moe (stub)")
    a = p.parse_args()
    return TrainConfig(**vars(a))


def main():
    cfg = parse_args()
    device = torch.device(cfg.device)
    # Data & tokenizer (HF only)
    tok: PreTrainedTokenizerBase
    from .data import hf_loader as _hf_loader
    train_loader, dev_loader, tok = _hf_loader.build_hf_dataloaders(
        cfg.task, cfg.dataset_size, cfg.seq_len, cfg.micro_batch
    )
    # Type hints for analyzers
    train_loader: DataLoader[Dict[str, torch.Tensor]] = train_loader
    dev_loader: DataLoader[Dict[str, torch.Tensor]] = dev_loader

    if cfg.backend == "tiny":
        vocab_sz: Optional[int] = len(tok.get_vocab()) if hasattr(tok, "get_vocab") else None
        model = make_model(cfg, vocab_size=vocab_sz).to(device)
    else:
        # HF backend: 現状は学習未対応。CausalLMで簡易サンプル生成のみ実施して終了。
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM

        model_name = "gpt2"
        tkn = AutoTokenizer.from_pretrained(model_name)  # type: ignore
        if tkn.pad_token is None:  # type: ignore
            tkn.pad_token = tkn.eos_token or tkn.unk_token  # type: ignore
        mdl = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # type: ignore
        mdl.eval()  # type: ignore
        prompt = "Hello from MurmNet!"  # 簡易デモ
        with torch.no_grad():
            enc = tkn(prompt, return_tensors="pt", add_special_tokens=False).to(device)  # type: ignore
            input_ids = enc["input_ids"]  # type: ignore
            for _ in range(32):
                out = mdl(input_ids)  # type: ignore
                next_logits = out.logits[:, -1, :]  # type: ignore
                next_id = next_logits.softmax(-1).multinomial(1)  # type: ignore
                input_ids = torch.cat([input_ids, next_id], dim=1)
                if tkn.eos_token_id is not None and int(next_id.item()) == int(tkn.eos_token_id):  # type: ignore
                    break
            text = tkn.decode(input_ids[0].tolist(), skip_special_tokens=True)  # type: ignore
        print("[hf_moe] sample:", text[len(prompt):])  # type: ignore
        raise SystemExit("hf_moe backend is inference-only for now. Use --backend tiny to train.")

    opt = AdamW(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    # Boids (router level)
    boids_router = BoidsRouterLoss(
        lambda_coh=cfg.boids_lambda_c,
        lambda_sep=cfg.boids_lambda_s,
        lambda_ali=cfg.boids_lambda_a,
        k_nn=cfg.boids_k_nn,
        sep_tau=cfg.boids_sep_tau,
    ) if cfg.boids_on else None
    boids_cfg: BoidsConfig = BoidsConfig(
        lambda_c=cfg.boids_lambda_c,
        lambda_s=cfg.boids_lambda_s,
        lambda_a=cfg.boids_lambda_a,
        k_nn=cfg.boids_k_nn,
        sep_tau=cfg.boids_sep_tau,
        warmup_frac=cfg.boids_warmup_frac,
    )
    boids = BoidsRegularizer(boids_cfg) if cfg.boids_on else None

    model.train()
    global_step = 0
    total_steps = cfg.train_epochs * len(train_loader)
    scaler = AmpGradScaler("cuda", enabled=torch.cuda.is_available())

    for epoch in range(cfg.train_epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # Mixed precision forward
            amp_ctx = amp_autocast(device_type="cuda", enabled=torch.cuda.is_available()) if torch.cuda.is_available() else nullcontext()
            with amp_ctx:
                out = model(input_ids, return_hidden=True)
                logits, util, hidden, moe_stats = cast(
                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]],
                    out,
                )
                # LM CE (shifted)
                bsz, tsz, vsz = logits.shape
                loss_ce = ce(logits.view(bsz * tsz, vsz), labels.view(bsz * tsz))
                # Switch Aux LB: sum over MoE layers of (num_experts * (f_e * P_e).sum())
                loss_moe = torch.tensor(0.0, device=device)
                if len(moe_stats) > 0:
                    for st in moe_stats:
                        f = st["f"]  # fraction routed by primary choice
                        P = st["P"]  # mean soft gate prob
                        loss_moe = loss_moe + cfg.load_balance_coef * (model.cfg.num_experts * (f * P).sum())
                else:
                    # Fallback proxy if no stats available
                    loss_moe = cfg.load_balance_coef * (util * util).sum()
                loss_boids = torch.tensor(0.0, device=device)
                if boids is not None:
                    boids.step(global_step, total_steps)
                    # Use last hidden states as features (compact)
                    loss_boids = boids(hidden)
                # Router-level Boids on gates if available (best-effort)
                loss_boids_router = torch.tensor(0.0, device=device)
                if boids_router is not None:
                    # Router-level Boids on actual last MoE layer gates
                    z = hidden.reshape(bsz * tsz, -1)
                    if len(moe_stats) > 0:
                        last = moe_stats[-1]
                        gates_soft = last["gates_soft"]  # (N,E)
                        topk_idx = last["topk_idx"]  # (N,K)
                    else:
                        gates_soft = torch.full((bsz * tsz, model.cfg.num_experts), 1.0 / model.cfg.num_experts, device=device)
                        topk_idx = torch.zeros(bsz * tsz, model.cfg.top_k, dtype=torch.long, device=device)
                    br = boids_router(z, gates_soft, topk_idx, model.cfg.num_experts)
                    loss_boids_router = br["boids"]
                loss: torch.Tensor = loss_ce + loss_moe + loss_boids + loss_boids_router

            scaler.scale(loss / cfg.accum_steps).backward()  # type: ignore[misc]
            if (it + 1) % cfg.accum_steps == 0:
                scaler.unscale_(opt)
                clip_grad_norm(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            global_step += 1
            pbar.set_postfix({"loss": float(loss.item()), "boids": float(loss_boids.item()) if boids else 0.0})  # type: ignore[arg-type]

    # Quick eval: self-consistency via multiple stochastic passes (dropout active)
    # Enable dropout for stochasticity during self-consistency measurement
    model.train()
    logits_runs: List[torch.Tensor] = []
    util_eval: Optional[torch.Tensor] = None
    last_batch: Optional[Dict[str, torch.Tensor]] = None
    with torch.no_grad():
        for _ in range(cfg.eval_trials):
            for batch in dev_loader:
                input_ids = batch["input_ids"].to(device)
                logits, util_eval = model(input_ids)
                logits_runs.append(logits.cpu())
                last_batch = batch
                break  # one batch per trial
    sc = self_consistency_rate(logits_runs)
    ent = expert_entropy(util_eval.detach().cpu()) if util_eval is not None else 0.0
    # Compute n-gram repetition, and task metrics (EM/F1/ROUGE/Acc) on last dev batch
    from .eval.metrics import squad_em_f1, rougeL_f1, sst2_label_from_text
    last_logits: torch.Tensor = logits_runs[-1]
    rep2: float = 0.0
    rep3: float = 0.0
    task_metric: Dict[str, float] = {}
    if last_batch is not None:
        labels_dev = last_batch["labels"]  # (B,T)
        greedy = last_logits.argmax(dim=-1)  # (B,T)
        pad_pred_ids: List[int] = list(map(int, greedy[0].tolist()))  # type: ignore[assignment]
        rep2 = ngram_repeat_rate(pad_pred_ids, 2)
        rep3 = ngram_repeat_rate(pad_pred_ids, 3)
        # compute per-sample metrics on answer spans (labels != -100)
        em_list: List[float] = []
        f1_list: List[float] = []
        rl_list: List[float] = []
        acc_list: List[float] = []
        for i in range(int(labels_dev.size(0))):
            mask = labels_dev[i] != -100
            if not bool(mask.any()):
                continue
            tgt_ids: List[int] = list(map(int, labels_dev[i][mask].tolist()))  # type: ignore[arg-type]
            prd_ids: List[int] = list(map(int, greedy[i][mask].tolist()))      # type: ignore[arg-type]
            tgt_txt = tok.decode(tgt_ids, skip_special_tokens=True)  # type: ignore[reportUnknownMemberType]
            prd_txt = tok.decode(prd_ids, skip_special_tokens=True)  # type: ignore[reportUnknownMemberType]
            if cfg.task == "squad":
                em, f1 = squad_em_f1(prd_txt, tgt_txt)
                em_list.append(em)
                f1_list.append(f1)
            elif cfg.task == "cnndm":
                rl = rougeL_f1(prd_txt, tgt_txt)
                rl_list.append(rl)
            elif cfg.task == "sst2":
                gt = sst2_label_from_text(tgt_txt)
                pd = sst2_label_from_text(prd_txt)
                if gt is not None and pd is not None:
                    acc_list.append(1.0 if gt == pd else 0.0)
        if cfg.task == "squad":
            task_metric = {
                "squad_em": float(sum(em_list) / max(1, len(em_list))),
                "squad_f1": float(sum(f1_list) / max(1, len(f1_list))),
            }
        elif cfg.task == "cnndm":
            task_metric = {"rougeL_f1": float(sum(rl_list) / max(1, len(rl_list)))}
        elif cfg.task == "sst2":
            task_metric = {"acc": float(sum(acc_list) / max(1, len(acc_list)))}
    over = overuse_rate(util_eval.detach().cpu()) if util_eval is not None else 0.0
    summary = {"self_consistency": sc, "expert_entropy": ent, "overuse_rate": over, "repeat_2gram": rep2, "repeat_3gram": rep3}
    summary.update(task_metric)
    print(summary)


if __name__ == "__main__":
    main()
