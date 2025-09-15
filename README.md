# MurmNet (HF Switch-Transformer 専用)

このリポジトリは Hugging Face Transformers の Google Switch-Transformer（例: `google/switch-base-16`）のみを用いた学習・評価・簡易チャットを提供します。TinyMoE/Boids 関連は廃止しました。

## セットアップ

依存関係をインストールしてください。

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 学習/評価

学習は HF モデルの損失（labels を渡す）で行います。SQuAD / CNN/DailyMail / SST-2 に対応。

例（SQuAD、小規模で1epoch）:

```powershell
python -m src.train --task squad --dataset_size small --seq_len 512 --train_epochs 1 --backend hf_moe --model_id google/switch-base-16
```

主なオプション:

- `--task [squad|cnndm|sst2]`
- `--dataset_size [small|full]`
- `--seq_len`, `--train_epochs`, `--micro_batch`, `--accum_steps`
- データサブサンプル: `--train_fraction`, `--eval_fraction`, `--max_train_samples`, `--max_eval_samples`
- チェックポイント保存: `--save_dir`; 評価のみ: `--eval_only --ckpt_path <dir>`
- ログ頻度: `--log_every N`

評価は `generate()` 後にタスク別指標（SQuAD: EM/F1, CNN/DM: ROUGE-L F1, SST-2: Accuracy）と繰り返し率（2/3-gram）を出力します。

## チャット UI

簡易Web UIで HF Switch-Transformer と対話できます。

```powershell
python -m src.app.chat
```

起動後: <http://127.0.0.1:7860/>

## 比較 UI（廃止）

TinyMoE/BOIDS 比較は廃止しました。`src/app/compare.py` は非推奨メッセージのみを表示します。

