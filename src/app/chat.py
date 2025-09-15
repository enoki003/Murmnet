import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Mapping, Protocol, cast
import importlib

import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from ..tools.hf_compat import auto_tokenizer_from_pretrained

from ..models.moe import MoEConfig, TinyMoETransformer


@dataclass
class ChatConfig:
    model_size: str = "small"      # tiny/small/base (see src/train.py MODEL_SIZES)
    num_experts: int = 8
    top_k: int = 1
    router_dropout: float = 0.1
    load_balance_coef: float = 0.0  # not used at inference
    seq_len: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.9
    max_new_tokens: int = 64


def build_model(cfg: ChatConfig, vocab_size: Optional[int] = None) -> TinyMoETransformer:
    # Minimal size mapping matching train.py
    sizes = {
        "tiny": dict(hidden_dim=384, ffn_dim=1536, num_layers=4),
        "small": dict(hidden_dim=768, ffn_dim=3072, num_layers=6),
        "base": dict(hidden_dim=1024, ffn_dim=4096, num_layers=8),
    }
    base = sizes[cfg.model_size]
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
        temperature=1.0,
    )
    return TinyMoETransformer(mcfg)


class _TokenizerDecodeLike(Protocol):
    def decode(self, token_ids: "torch.Tensor | List[int] | int", *, skip_special_tokens: bool = ...) -> str: ...


def generate(model: TinyMoETransformer, tok: PreTrainedTokenizerBase, prompt: str, cfg: ChatConfig) -> str:
    device = torch.device(cfg.device)
    model.eval()
    with torch.no_grad():
        enc = cast(Mapping[str, torch.Tensor], tok(prompt, return_tensors="pt", add_special_tokens=False))
        input_ids = enc["input_ids"].to(device)
        # truncate last seq_len tokens to fit context
        input_ids = input_ids[:, -cfg.seq_len:]
        for _ in range(cfg.max_new_tokens):
            logits, _ = model(input_ids)
            next_logits = logits[:, -1, :] / max(1e-6, cfg.temperature)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            eos_id = cast(Optional[int], getattr(tok, 'eos_token_id', None))
            if eos_id is not None and next_id.item() == eos_id:
                break
    row_t = input_ids[0].to(dtype=torch.long)
    tok_dec = cast(_TokenizerDecodeLike, tok)
    out: str = tok_dec.decode(row_t, skip_special_tokens=True)
    return out[len(prompt):]


# --------- Gradio UI ---------
## Gradio is imported dynamically to avoid type stub issues


class _BlocksLike(Protocol):
    def launch(self, *, server_name: str, server_port: int, share: bool) -> object: ...


def build_demo():
    cfg = ChatConfig()
    tok_name = os.environ.get("MURMNET_TOKENIZER", "gpt2")
    tok: PreTrainedTokenizerBase = auto_tokenizer_from_pretrained(tok_name)
    if getattr(tok, "pad_token", None) is None:
        # Prefer EOS/UNK/SEP as padding if available
        pad_tok = getattr(tok, "eos_token", None) or getattr(tok, "unk_token", None) or getattr(tok, "sep_token", None)
        if pad_tok is not None:
            setattr(tok, "pad_token", pad_tok)
    vocab_size = int(getattr(tok, "vocab_size", 32000))
    model = build_model(cfg, vocab_size=vocab_size)
    model.to(cfg.device)

    gr = importlib.import_module("gradio")

    with gr.Blocks(title="MurmNet TinyMoE Chat") as demo:
        gr.Markdown("# MurmNet TinyMoE Chat\n小型MoEトランスフォーマでローカル会話")
        chatbot = gr.Chatbot(type="messages", height=400)
        with gr.Row():
            msg = gr.Textbox(label="メッセージ", scale=4)
            send = gr.Button("送信", variant="primary", scale=1)
        temperature = gr.Slider(0.2, 1.5, value=cfg.temperature, step=0.05, label="Temperature")
        max_new = gr.Slider(8, 256, value=cfg.max_new_tokens, step=8, label="Max new tokens")
        state = gr.State({"cfg": cfg, "tok": tok, "model": model})

        # define callback inside Blocks context
        def respond(
            user_message: str,
            chat_history: Optional[List[Dict[str, str]]],
            temperature: float,
            max_new: int,
            st: Dict[str, object],
        ) -> Tuple[List[Dict[str, str]], Dict[str, object]]:
            st_cfg = cast(ChatConfig, st["cfg"])  # runtime state typing
            st_tok = cast(PreTrainedTokenizerBase, st["tok"])  # tokenizer
            st_model = cast(TinyMoETransformer, st["model"])  # model
            # apply runtime knobs
            st_cfg.temperature = float(temperature)
            st_cfg.max_new_tokens = int(max_new)
            # build transcript
            lines: List[str] = []
            for m in chat_history or []:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    lines.append(f"User: {content}")
                elif role == "assistant":
                    lines.append(f"Assistant: {content}")
            lines.append(f"User: {user_message}")
            lines.append("Assistant: ")
            prompt = "\n".join(lines)
            out = generate(st_model, st_tok, prompt, st_cfg)
            new_history = (chat_history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": out},
            ]
            return new_history, st

        # bind events inside Blocks context
        send.click(respond, inputs=[msg, chatbot, temperature, max_new, state], outputs=[chatbot, state])
        msg.submit(respond, inputs=[msg, chatbot, temperature, max_new, state], outputs=[chatbot, state])

    return cast(_BlocksLike, demo)


def main():
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
