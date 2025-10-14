# MurmNet（Plan‑B: HF Switch + Boids 正則化）

このリポジトリは Hugging Face Transformers の Google Switch‑Transformer（例: `google/switch-base-16`）を用いた学習・評価・チャットを提供します。
Plan‑B として、Switch のルーティング確率に対して Boids 風正則化（整列・分散・任意のエントロピー）を必須で適用します。

※このリポジトリは制作中です

## セットアップ

依存関係をインストールしてください。

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 学習/評価

学習は HF モデルの損失（labels を渡す）で行います。SQuAD / CNN/DailyMail / SST‑2 に対応。

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

### Boids 正則化（必須）

Switch の `router_logits` を用いて Boids 正則化を必ず適用します。

```powershell
python -m src.train --task squad --dataset_size small --seq_len 256 --train_epochs 1 --model_id google/switch-base-16 ^
	--boids_on true --boids_weight 0.01 --boids_align 1.0 --boids_sep 1.0 --boids_entropy 0.0
```

- `--boids_on true`（デフォルト、必須）
- `--boids_weight` は損失への寄与係数
- `--boids_align`（隣接トークン整合）, `--boids_sep`（ロードバランス）, `--boids_entropy`（任意）

注意: モデルが `router_logits` を返さない場合はエラーになります（Switch 系モデルを使用してください）。

評価は `generate()` 後にタスク別指標（SQuAD: EM/F1, CNN/DM: ROUGE-L F1, SST-2: Accuracy）と繰り返し率（2/3-gram）を出力します。

## チャット UI

簡易Web UIで HF Switch-Transformer と対話できます。

```powershell
python -m src.app.chat
```

起動後: <http://127.0.0.1:7860/>

### 比較チャット UI（ベース vs チューニング済み）

同じ入力に対する応答を左右に並べて比較できます。

```powershell
python -m src.app.compare
```

起動後: <http://127.0.0.1:7861/>

- 左: ベースモデル（デフォルト `google/switch-base-16`）
- 右: `runs/` 配下のチェックポイントをドロップダウンから選択し、「読み込み/更新」を押下
- 「候補を再スキャン」で `runs/` のチェックポイント一覧を更新

## MoE 実装について

現在は HF Switch‑Transformer を前提にしています。専用の自前 MoE レイヤは同梱していませんが、
将来的にカスタム MoE 実装を追加する場合は、ルータ確率を `(B,T,E)` で各層リストとして `outputs.router_logits` 互換で提供すれば、
本リポジトリの Boids 正則化をそのまま適用できます。

## 今後のチャット対応予定

- チャットUIの拡張（システムプロンプト・会話テンプレート、履歴の保存/読み込み）
- チューニング済みチェックポイントの直接指定や選択UXの改善（チャット単体でも選択可）
- 生成パラメータ（温度・反復回避など）の詳細設定、CPU/GPU切り替えの明示化

