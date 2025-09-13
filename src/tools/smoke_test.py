import torch
from ..models.moe import TinyMoETransformer, MoEConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MoEConfig(hidden_dim=384, ffn_dim=1536, num_layers=2, num_experts=4, top_k=1)
    model = TinyMoETransformer(cfg).to(device)
    x = torch.randint(0, 32000, (2, 32), device=device)
    out, util, hidden, moe_stats = model(x, return_hidden=True)
    # Use a small piece of stats to ensure tensors are present
    stats_term = moe_stats[-1]["P"].mean() if len(moe_stats) > 0 else torch.tensor(0.0, device=device)
    loss = out.mean() + util.mean() + hidden.mean() + stats_term
    loss.backward()
    print("smoke ok", out.shape, util.shape)


if __name__ == "__main__":
    main()
