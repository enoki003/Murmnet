import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

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


def generate(model: TinyMoETransformer, tok, prompt: str, cfg: ChatConfig) -> str:
    device = torch.device(cfg.device)
    model.eval()
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        # truncate last seq_len tokens to fit context
        input_ids = input_ids[:, -cfg.seq_len:]
        for _ in range(cfg.max_new_tokens):
            logits, _ = model(input_ids)
            next_logits = logits[:, -1, :] / max(1e-6, cfg.temperature)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if hasattr(tok, 'eos_token_id') and tok.eos_token_id is not None and next_id.item() == tok.eos_token_id:
                break
        out = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)
        return out[len(prompt):]


# --------- Gradio UI ---------
import gradio as gr


def build_demo():
    cfg = ChatConfig()
    tok_name = os.environ.get("MURMNET_TOKENIZER", "gpt2")
    tok = AutoTokenizer.from_pretrained(tok_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token or tok.sep_token

    model = build_model(cfg, vocab_size=len(tok))
    model.to(cfg.device)

    with gr.Blocks(title="MurmNet TinyMoE Chat") as demo:
        gr.Markdown("# MurmNet TinyMoE Chat\n小型MoEトランスフォーマでローカル会話")
        chatbot = gr.Chatbot(type="messages", height=400)
        with gr.Row():
            msg = gr.Textbox(label="メッセージ", scale=4)
            send = gr.Button("送信", variant="primary", scale=1)
        temperature = gr.Slider(0.2, 1.5, value=cfg.temperature, step=0.05, label="Temperature")
        max_new = gr.Slider(8, 256, value=cfg.max_new_tokens, step=8, label="Max new tokens")

        state = gr.State({"cfg": cfg, "tok": tok, "model": model})

        def respond(user_message, chat_history, temperature, max_new, st):
            st_cfg: ChatConfig = st["cfg"]
            st_tok = st["tok"]
            st_model: TinyMoETransformer = st["model"]
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

        send.click(respond, inputs=[msg, chatbot, temperature, max_new, state], outputs=[chatbot, state])
        msg.submit(respond, inputs=[msg, chatbot, temperature, max_new, state], outputs=[chatbot, state])

    return demo


def main():
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
