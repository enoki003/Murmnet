from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Protocol, cast, Any, Mapping
import importlib

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class ChatConfig:
    model_id: str = "google/switch-base-16"
    seq_len: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.9
    max_new_tokens: int = 64

def build_hf(model_id: str, device: torch.device) -> Tuple[PreTrainedTokenizerBase, Any]:
    tok_mod = importlib.import_module("transformers.models.auto.tokenization_auto")
    AutoTokenizer = getattr(tok_mod, "AutoTokenizer")
    tok: Any = AutoTokenizer.from_pretrained(model_id)
    if getattr(tok, "pad_token", None) is None:
        pad_tok = getattr(tok, "eos_token", None) or getattr(tok, "unk_token", None) or getattr(tok, "sep_token", None)
        if pad_tok is not None:
            setattr(tok, "pad_token", pad_tok)
    mdl_mod = importlib.import_module("transformers.models.auto.modeling_auto")
    AutoModelForSeq2SeqLM = getattr(mdl_mod, "AutoModelForSeq2SeqLM")
    mdl_any: Any = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    mdl = mdl_any.to(device)
    return cast(PreTrainedTokenizerBase, tok), mdl


class _TokenizerDecodeLike(Protocol):
    def decode(self, token_ids: "torch.Tensor | List[int] | int", *, skip_special_tokens: bool = ...) -> str: ...


def generate(model: Any, tok: PreTrainedTokenizerBase, prompt: str, cfg: ChatConfig) -> str:
    device = torch.device(cfg.device)
    model.eval()
    with torch.no_grad():
        enc_any = tok(prompt, return_tensors="pt", add_special_tokens=True)
        enc = cast(Mapping[str, Any], enc_any)
        input_ids = cast(torch.Tensor, enc["input_ids"]).to(device)
        input_ids = input_ids[:, -cfg.seq_len:]
    gen = model.generate(input_ids=input_ids, max_new_tokens=cfg.max_new_tokens, do_sample=False)
    out = cast(_TokenizerDecodeLike, tok).decode(gen[0], skip_special_tokens=True)
    return out


# --------- Gradio UI ---------
## Gradio is imported dynamically to avoid type stub issues


class _BlocksLike(Protocol):
    def launch(self, *, server_name: str, server_port: int, share: bool) -> object: ...


def build_demo():
    cfg = ChatConfig()
    tok, model = build_hf(cfg.model_id, torch.device(cfg.device))

    gr = importlib.import_module("gradio")

    with gr.Blocks(title="Switch Transformer Chat") as demo:
        gr.Markdown("# Switch Transformer Chat\nHF google/switch-base-16 でローカル会話")
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
            st_model = st["model"]  # HF model
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
            lines.append("Assistant:")
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
