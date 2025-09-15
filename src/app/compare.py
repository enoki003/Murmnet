import importlib


def main() -> None:
    gr = importlib.import_module("gradio")
    with gr.Blocks(title="Compare (deprecated)") as demo:
        gr.Markdown(
            """
            # Compare UI (deprecated)
            TinyMoE と BOIDS 比較UIは廃止しました。現在は Hugging Face の Switch-Transformer のみをサポートしています。\n
            - 学習/評価: `python -m src.train --backend hf_moe`
            - チャット: `python -m src.app.chat`
            """
        )
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
