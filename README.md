# Plan-B: MoE Router + Boids Regularization（まず回す版）

このリポジトリは、Plan-B（MoEルータ＋Boids正則化）を“まず回す”ための最小〜推奨設定を含む学習フレームです。

- 対応データ: SQuAD v1.1/2.0, CNN/DailyMail, SST-2（GLUE）
- 目的関数: 既存タスク損失（CE/LM）＋ MoEロードバランス損失 ＋ Boids正則化（C/S/A）
- 目標: Boids有無で安定性・ロードバランス・品質に差が出ることを初期スケールで確認

## クイックスタート（スモークテスト）

依存関係をインストール後、ダミーデータで1ステップ動作確認します。

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.tools.smoke_test
```

成功すれば、MoE層＋Boids項を含む forward/backward が1バッチ通ります。

## まず回す実行例（HFデータローダ: small/full）

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.train \
  --task squad \
  --dataset_size small \
  --model_size small \
  --num_experts 8 --top_k 1 \
  --seq_len 512 \
  --train_epochs 1 \
  --boids_on true \
  --boids_lambda_c 0.1 --boids_lambda_s 0.05 --boids_lambda_a 0.01 \
  --router_dropout 0.1 --load_balance_coef 0.01 \
  --eval_trials 5
```

CNN/DailyMailやSST-2に切替える場合（small=小規模サブセット）:

```powershell
# 要約（LM目的で記事本文を自己回帰学習）
python -m src.train --task cnndm --dataset_size small --model_size small --seq_len 1024 --train_epochs 1 --boids_on true

# 短文分類（単純CE）
python -m src.train --task sst2 --dataset_size small --model_size tiny --seq_len 256 --train_epochs 1 --boids_on true
```

メモ:

- 本リポはWindows対応を考慮し、DataLoaderのワーカ数は既定で0に設定しています（マルチプロセスのpickle問題回避）。
- `--dataset_size` は `small`/`full`。ダミーデータは廃止しました。

## バックエンド切替（学習: tiny / 推論デモ: hf_moe）

- 既定の学習バックエンドは本リポの TinyMoE（`--backend tiny`）。
- 外部HFモデル（例: gpt2）を使った簡易推論デモは `--backend hf_moe` で動きます（学習は終了し、サンプル出力のみ）。

```powershell
python -m src.train --backend hf_moe
```

上記は1サンプルを生成して終了します。学習は `--backend tiny` を使用してください。

## ローカルWebチャット（TinyMoE）

簡易Web UIでTinyMoEと対話できます。

```powershell
python -m src.app.chat
```

ローカルURL: <http://127.0.0.1:7860>

## 三条件比較（通常 / BOIDS / なし）

研究用に、以下の3モードを同一条件で順に走らせ、最後にJSONで比較結果を出力するユーティリティを用意しています。

- none: 補助損失なし（load balance=0, boids off）
- normal: MoEロードバランスのみ（boids off）
- boids: ロードバランス＋Boids正則化

実行（PowerShell）:

```powershell
python -m src.tools.compare_modes --preset fast   # 速い比較（tiny/4 experts/seq 128, 1 epoch）
python -m src.tools.compare_modes --preset full   # 推奨スケール（small/8 experts/seq 512, 1 epoch）
```

備考:

- 各モードは `src.train` をサブプロセスで実行し、終了時に `src/train.py` が出力する最終JSONを収集します。
- 3ジョブは順番に実行されます（同時ではありません）。別の学習タスクを同時に走らせている場合は、先にそちらを止めるか、比較ユーティリティの実行を後にしてください。
- 乱数は `--seed 42` 固定で呼び出しています。個別に変更したい場合は `src/tools/compare_modes.py` の `base` 引数を調整してください。

## 設定プリセット（規模別の現実的ライン）

- 個人1GPU（24GB級）: `--model_size small --num_experts 8 --top_k 1 --seq_len 512 --micro_batch 1 --accum_steps 16`
- 2GPU（40–48GB×2）: `--model_size base --num_experts 16 --top_k 2 --seq_len 1024 --micro_batch 2 --accum_steps 16`

詳細は `configs/presets.json` を参照。

## 重要ハイパラ（初期値）

- MoE: `num_experts=16, top_k=2, router_dropout=0.1, load_balance_coef=0.01`
- Boids: `lambda_c=0.1, lambda_s=0.05, lambda_a=0.01, k_nn=8, sep_tau=1.5`
- Boidsウォームアップ: 前半の `--boids_warmup_frac` で 0→既定値へ線形上昇

## 評価指標（まず回す版）

- 安定: 出力自己一致率、埋め込み分散
- MoE効率: expert利用エントロピー、偏り超過率
- 品質: EM/F1（SQuAD）、ROUGE-L F1（CNN/DM）、Accuracy（SST-2）
  - devバッチ1個を貪欲デコードで簡易評価して平均を表示します（`src/train.py` 終了時にJSONで出力）。
- 冗長: n-gram反復率（2–4gram）

## 注意

- WindowsでGPU学習を行う場合、PyTorchはCUDA対応版の事前インストールが必要です（公式手順参照）。
- DeepSpeed/FSDPはオプションです（まず回す版は標準PyTorchで動作）。

## 再現性の固定（推奨）

- `torch.manual_seed`, `random.seed`, `numpy.random.seed`
- `torch.backends.cudnn.deterministic=True` と `torch.use_deterministic_algorithms(True)`
- DataLoaderの`worker_init_fn`で各ワーカseedを固定
- `PYTHONHASHSEED`を環境変数で固定
- ログに `git commit`, `有効cfgのSHA`, `環境情報` を保存

## 外部モデル・データセットの調達計画（参考）

- モデル（MoE系の例）
  - Switch Transformer系（HF上で公開のものはFlax実装が中心です。PyTorchでの学習は非対応のことが多いので要確認）。
  - Qwen2-MoEなどの軽量MoE（PyTorchでの微調整が可能な公開モデルがある場合はそちらを推奨）。
  - Mixtral等の大規模MoEは個人GPUでは非現実的（推奨外）。
  - ライセンスは各モデルのHFページで必ずご確認ください（研究目的/商用可否が分かれます）。

- データセット
  - SQuAD（QA）、CNN/DailyMail（要約）、SST-2（感情分類）を `datasets` から取得可能。
  - それぞれのライセンスはHFページでの最新情報を確認してください（SQuADやCNN/DMはCC系、GLUEはタスク毎に条件が異なります）。

### 調達スクリプト

最低限のモデル・データセットのキャッシュ取得は以下で可能です。

```powershell
python -m src.tools.procure_assets
```

> 備考: 本リポの学習ループは“TinyMoE”バックエンドが既定です。外部MoE（Switch/Qwen2-MoE等）による学習は将来の拡張で提供予定です。

## ライセンス・法的注意

- 本リポジトリのコードライセンスは未確定です（プロジェクト所有者による選択を想定）。
- `LEGAL.md` に第三者資産（Hugging Faceモデル/データセット等）に関する注意事項を記載しています。各資産のライセンス・利用条件を遵守してください。

