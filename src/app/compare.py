from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Protocol, cast, Any, Mapping
import importlib
import os

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# --------- Core helpers (shared with chat.py style) ---------


@dataclass
class ChatConfig:
    model_id: str = "google/switch-base-16"
    seq_len: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.9
    max_new_tokens: int = 64


def build_hf(model_id_or_path: str, device: torch.device) -> Tuple[PreTrainedTokenizerBase, Any]:
    tok_mod = importlib.import_module("transformers.models.auto.tokenization_auto")
    AutoTokenizer = getattr(tok_mod, "AutoTokenizer")
    tok: Any = AutoTokenizer.from_pretrained(model_id_or_path)
    if getattr(tok, "pad_token", None) is None:
        pad_tok = getattr(tok, "eos_token", None) or getattr(tok, "unk_token", None) or getattr(tok, "sep_token", None)
        if pad_tok is not None:
            setattr(tok, "pad_token", pad_tok)
    mdl_mod = importlib.import_module("transformers.models.auto.modeling_auto")
    AutoModelForSeq2SeqLM = getattr(mdl_mod, "AutoModelForSeq2SeqLM")
    mdl_any: Any = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path)
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


# --------- Utilities ---------


def list_tuned_checkpoints(runs_dir: str = "runs") -> List[str]:
    candidates: List[str] = []
    if not os.path.isdir(runs_dir):
        return candidates
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if not os.path.isdir(path):
            continue
        # Heuristic: presence of config.json and model.safetensors
        if os.path.isfile(os.path.join(path, "config.json")) and any(
            os.path.isfile(os.path.join(path, fn)) for fn in ("pytorch_model.bin", "model.safetensors")
        ):
            candidates.append(path)
    candidates.sort()
    return candidates


# --------- Gradio UI ---------


class _BlocksLike(Protocol):
    def launch(self, *, server_name: str, server_port: int, share: bool) -> object: ...


def build_demo():
    cfg = ChatConfig()
    base_tok, base_model = build_hf(cfg.model_id, torch.device(cfg.device))

    gr = importlib.import_module("gradio")

    tuned_options = list_tuned_checkpoints("runs")
    tuned_default = tuned_options[-1] if tuned_options else ""

    with gr.Blocks(title="Compare: Base vs Tuned") as demo:
        gr.Markdown("# 比較チャット: ベース vs チューニング済み\n同じ入力に対する応答を並べて比較できます。右側は runs/ 配下から選択します。")

        with gr.Row():
            tuned_dropdown = gr.Dropdown(
                label="チューニング済みチェックポイント (runs/...)",
                choices=(tuned_options if tuned_options else ["(runs/ に候補がありません)"]),
                value=(tuned_default if tuned_default else None),
                allow_custom_value=True,
                interactive=True,
            )
            load_btn = gr.Button("読み込み/更新", variant="primary")
            refresh_btn = gr.Button("候補を再スキャン")
            status = gr.Markdown("")
        with gr.Row():
            manual_path = gr.Textbox(label="手動パス入力 (例: runs/switch-base-16_squad_small_boids)", placeholder="runs/...", scale=4)
            manual_load = gr.Button("手動読み込み")

        temperature = gr.Slider(0.2, 1.5, value=cfg.temperature, step=0.05, label="Temperature (共通)")
        max_new = gr.Slider(8, 256, value=cfg.max_new_tokens, step=8, label="Max new tokens (共通)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 左: ベース (google/switch-base-16)")
                chat_base = gr.Chatbot(type="messages", height=380)
            with gr.Column():
                gr.Markdown("### 右: チューニング済み")
                chat_tuned = gr.Chatbot(type="messages", height=380)

        with gr.Row():
            msg = gr.Textbox(label="メッセージ", scale=4)
            send = gr.Button("両方に送信", variant="primary", scale=1)
            clear = gr.Button("クリア")

        # Runtime state
        state = gr.State({
            "cfg": cfg,
            "base_tok": base_tok,
            "base_model": base_model,
            "tuned_tok": None,
            "tuned_model": None,
            "tuned_id": tuned_default,
        })

        def do_refresh():
            opts = list_tuned_checkpoints("runs")
            if not opts:
                return gr.update(choices=["(runs/ に候補がありません)"], value=None), "runs/ 配下にチェックポイントが見つかりませんでした。学習後に再試行してください。"
            return gr.update(choices=opts, value=opts[-1]), f"候補を更新しました (合計 {len(opts)} 件)"

        def do_load(sel: str, st: Dict[str, object]) -> Tuple[Dict[str, object], str]:
            sel = (sel or "").strip()
            if not sel:
                # Keep tuned unset; user may only use base
                st["tuned_tok"] = None
                st["tuned_model"] = None
                st["tuned_id"] = ""
                return st, "右側は未選択のため空のままです。"
            device = torch.device(cast(ChatConfig, st["cfg"]).device)
            tok, mdl = build_hf(sel, device)
            st["tuned_tok"] = tok
            st["tuned_model"] = mdl
            st["tuned_id"] = sel
            return st, f"読み込み完了: {sel}"

        def respond(
            user_message: str,
            base_history: Optional[List[Dict[str, str]]],
            tuned_history: Optional[List[Dict[str, str]]],
            temperature_val: float,
            max_new_val: int,
            st: Dict[str, object],
        ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, object]]:
            st_cfg = cast(ChatConfig, st["cfg"])
            st_cfg.temperature = float(temperature_val)
            st_cfg.max_new_tokens = int(max_new_val)

            # Shared transcript builder
            def build_prompt(hist: Optional[List[Dict[str, str]]], message: str) -> str:
                lines: List[str] = []
                for m in hist or []:
                    role = m.get("role")
                    content = m.get("content", "")
                    if role == "user":
                        lines.append(f"User: {content}")
                    elif role == "assistant":
                        lines.append(f"Assistant: {content}")
                lines.append(f"User: {message}")
                lines.append("Assistant:")
                return "\n".join(lines)

            # Base inference
            base_tok_loc = cast(PreTrainedTokenizerBase, st["base_tok"])  # type: ignore
            base_model_loc = st["base_model"]
            base_out = generate(base_model_loc, base_tok_loc, build_prompt(base_history, user_message), st_cfg)
            new_base = (base_history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": base_out},
            ]

            # Tuned inference (if loaded)
            tuned_tok_loc = cast(Optional[PreTrainedTokenizerBase], st.get("tuned_tok"))
            tuned_model_loc = st.get("tuned_model")
            if tuned_tok_loc is not None and tuned_model_loc is not None:
                tuned_out = generate(tuned_model_loc, tuned_tok_loc, build_prompt(tuned_history, user_message), st_cfg)
                new_tuned = (tuned_history or []) + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": tuned_out},
                ]
            else:
                # If not loaded, echo an instruction
                new_tuned = (tuned_history or []) + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": "(右側は未選択です。上のドロップダウンから runs/ のチェックポイントを選び、読み込みを押してください)"},
                ]

            return new_base, new_tuned, st

        # Wire events (inside Blocks context)
        refresh_btn.click(do_refresh, inputs=[], outputs=[tuned_dropdown, status])
        load_btn.click(do_load, inputs=[tuned_dropdown, state], outputs=[state, status])

        def do_manual_load(path: str, st: Dict[str, object]) -> Tuple[Dict[str, object], str]:
            return do_load(path, st)

        manual_load.click(do_manual_load, inputs=[manual_path, state], outputs=[state, status])

        send.click(
            respond,
            inputs=[msg, chat_base, chat_tuned, temperature, max_new, state],
            outputs=[chat_base, chat_tuned, state],
        )
        msg.submit(
            respond,
            inputs=[msg, chat_base, chat_tuned, temperature, max_new, state],
            outputs=[chat_base, chat_tuned, state],
        )
        def do_clear() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
            return [], [], ""

        clear.click(do_clear, inputs=[], outputs=[chat_base, chat_tuned, status])

    return cast(_BlocksLike, demo)


def main() -> None:
    demo = build_demo()
    # Use 7861 to avoid clashing with chat.py (7860)
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
