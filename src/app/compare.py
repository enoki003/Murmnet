import json
import subprocess
import sys
from typing import Any, Dict, Optional
import importlib
import re


def run_compare(preset: str, train_fraction: Optional[float], eval_fraction: Optional[float]) -> Dict[str, Any]:
    args = [sys.executable, "-m", "src.tools.compare_modes", "--preset", preset]
    if train_fraction is not None:
        args += ["--train_fraction", str(train_fraction)]
    if eval_fraction is not None:
        args += ["--eval_fraction", str(eval_fraction)]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    # Extract the last JSON object (multi-line supported)
    parsed: Optional[Dict[str, Any]] = None
    m = re.search(r"\{[\s\S]*\}\s*$", out)
    if m:
        block = m.group(0)
        try:
            parsed = json.loads(block)
        except Exception:
            parsed = None
    if parsed is None:
        # Fallback: find last line starting with '{' and join to the end
        lines = out.splitlines()
        start = -1
        for i, ln in enumerate(lines):
            if ln.strip().startswith("{"):
                start = i
        if start != -1:
            chunk = "\n".join(lines[start:])
            try:
                parsed = json.loads(chunk)
            except Exception:
                parsed = None
    if parsed is None:
        raise RuntimeError("Failed to parse comparison output.\n\n" + out)
    return parsed


def main() -> None:
    gr = importlib.import_module("gradio")
    with gr.Blocks(title="MurmNet Compare Modes") as demo:
        gr.Markdown("# Compare: none vs normal vs boids\nサブサンプリングで素早く傾向を確認")
        with gr.Row():
            preset = gr.Radio(["fast", "full"], value="fast", label="Preset")
            train_frac = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Train fraction")
            eval_frac = gr.Slider(0.01, 1.0, value=0.2, step=0.01, label="Eval fraction")
        run_btn = gr.Button("Run comparison", variant="primary")
        out_json = gr.JSON(label="Results JSON")

        def _run(preset_val: str, trf: float, evf: float):
            res = run_compare(preset_val, trf, evf)
            return res

        run_btn.click(_run, inputs=[preset, train_frac, eval_frac], outputs=[out_json])

    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)


if __name__ == "__main__":
    main()
