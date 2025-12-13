# analyze_geometry.py
import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pyarrow.parquet as pq


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count() or 4


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class BatchTopKSAE(torch.nn.Module):
    def __init__(self, d_in: int, m: int, target_k: int):
        super().__init__()
        self.d_in = d_in
        self.m = m
        self.target_k = target_k

        self.encoder = torch.nn.Linear(d_in, m, bias=True)
        self.decoder = torch.nn.Linear(m, d_in, bias=False)

    def batch_topk(self, a: torch.Tensor) -> torch.Tensor:
        B, m = a.shape
        total_k = min(self.target_k * B, B * m)
        if total_k == B * m:
            return a
        flat = a.view(-1)
        _, idx = torch.topk(flat, k=total_k, largest=True)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        mask = mask.view(B, m)
        return a * mask

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        a = torch.relu(z)
        a = self.batch_topk(a)
        return z, a

    def decode(self, codes: torch.Tensor):
        return self.decoder(codes)


def load_embeddings(cfg):
    emb_dir = cfg["extract"].get("output_dir", "./outputs")
    emb_filename = cfg["extract"].get("embeddings_filename", "embeddings.parquet")
    path = os.path.join(emb_dir, emb_filename)

    table = pq.read_table(path)
    df = table.to_pandas()

    img_emb = torch.tensor(
        list(df["image_embedding"].values), dtype=torch.float32
    )  # [N, d]
    txt_emb = torch.tensor(
        list(df["text_embedding"].values), dtype=torch.float32
    )  # [N, d]
    captions = df["caption"].tolist()

    return img_emb, txt_emb, captions


def load_sae(cfg):
    sae_cfg = cfg["sae"]
    sae_dir = sae_cfg.get("sae_dir", "./outputs/sae")
    ckpt_path = os.path.join(sae_dir, "sae_checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    d_in = ckpt["d_in"]
    code_dim = ckpt["code_dim"]
    target_k = ckpt["target_k"]

    sae = BatchTopKSAE(d_in=d_in, m=code_dim, target_k=target_k)
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(DEVICE)
    sae.eval()
    return sae, d_in, code_dim, target_k


def encode_all_codes(sae, X: torch.Tensor, batch_size: int = 1024):
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    codes_list = []
    with torch.no_grad():
        for (x_batch,) in dl:
            x_batch = x_batch.to(DEVICE)
            _, codes = sae.encode(x_batch)
            codes_list.append(codes.cpu())
    codes = torch.cat(codes_list, dim=0)
    return codes  # [N, m]


def concept_energy(codes: torch.Tensor, squared: bool = True) -> torch.Tensor:
    if squared:
        vals = codes ** 2
    else:
        vals = codes.abs()
    return vals.mean(dim=0)  # [m]


def modality_score(codes_img: torch.Tensor, codes_txt: torch.Tensor) -> torch.Tensor:
    e_img = concept_energy(codes_img, squared=True)
    e_txt = concept_energy(codes_txt, squared=True)
    denom = e_img + e_txt + 1e-8
    return e_img / denom  # [m] in [0,1]


def bridge_matrix(codes_img, codes_txt, dictionary):
    """
    B = E[c_img ⊗ c_txt] ⊙ S
    """
    assert codes_img.shape == codes_txt.shape
    N, m = codes_img.shape
    C = (codes_img.T @ codes_txt) / float(N)  # [m, m]

    D = dictionary
    D = D / (D.norm(dim=-1, keepdim=True) + 1e-8)
    S = D @ D.T  # [m, m]

    return (C * S).cpu().numpy()


def analyze_geometry(config_path: str):
    cfg = load_config(config_path)

    img_emb, txt_emb, _ = load_embeddings(cfg)
    sae, d_in, code_dim, target_k = load_sae(cfg)

    print(f"[analysis] Embeddings img {img_emb.shape}, txt {txt_emb.shape}")
    print(f"[analysis] SAE code_dim={code_dim}, target_k={target_k}")

    # Compute codes
    codes_img = encode_all_codes(sae, img_emb)
    codes_txt = encode_all_codes(sae, txt_emb)

    # Energy
    energy_all = concept_energy(torch.cat([codes_img, codes_txt], dim=0))  # [m]

    # Modality score
    mod_score = modality_score(codes_img, codes_txt)  # [m]

    # Bridge matrix
    dictionary = sae.decoder.weight.detach().cpu().T  # [m, d]
    B = bridge_matrix(codes_img, codes_txt, dictionary)

    metrics_dir = cfg["analysis"].get("metrics_dir", "./outputs/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    np.savez(
        os.path.join(metrics_dir, "metrics.npz"),
        energy=energy_all.cpu().numpy(),
        modality_score=mod_score.cpu().numpy(),
        bridge_matrix=B,
    )
    print(f"[analysis] Saved metrics (energy, modality, bridge) to {metrics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    analyze_geometry(args.config)
