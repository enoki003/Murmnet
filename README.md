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

## まず回す実行例（SQuADサブセット相当: dummy/small）

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.train \
  --task squad \
  --dataset_size dummy \
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
- 品質: EM/F1（SQuAD）、ROUGE（CNN/DM）、Accuracy（SST-2）
  - 本リリースでは分類Accuracyを即時提供、EM/ROUGEは後述の簡易評価器で参照可
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

> 備考: 本リポの学習ループは“TinyMoE”バックエンドが既定です。外部MoE（Switch/Qwen2-MoE等）による学習は将来の拡張で提供予定です。
