# train_sae.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import pyarrow.parquet as pq


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = os.cpu_count() or 4


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class BatchTopKSAE(nn.Module):
    """
    BatchTopK Sparse Autoencoder.

    x: [B, d]
    encoder: Linear -> [B, m]
    relu, then BatchTopK over batch (top K * B activations)
    decoder: [m, d] dictionary atoms (rows)
    """

    def __init__(self, d_in: int, m: int, target_k: int):
        super().__init__()
        self.d_in = d_in
        self.m = m
        self.target_k = target_k

        self.encoder = nn.Linear(d_in, m, bias=True)
        self.decoder = nn.Linear(m, d_in, bias=False)

        nn.init.kaiming_normal_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=-1
            )

    def batch_topk(self, a: torch.Tensor) -> torch.Tensor:
        B, m = a.shape
        total_k = min(self.target_k * B, B * m)
        if total_k == B * m:
            return a
        flat = a.view(-1)
        vals, idx = torch.topk(flat, k=total_k, largest=True)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        mask = mask.view(B, m)
        return a * mask

    def normalize_dictionary(self):
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=-1
            )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        a = torch.relu(z)
        a = self.batch_topk(a)
        x_hat = self.decoder(a)
        return x_hat, z, a

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
    ) 
    txt_emb = torch.tensor(
        list(df["text_embedding"].values), dtype=torch.float32
    ) 

    return img_emb, txt_emb, df["caption"].tolist()


def train_sae(config_path: str):
    cfg = load_config(config_path)
    sae_cfg = cfg["sae"]

    img_emb, txt_emb, _ = load_embeddings(cfg)
    X = torch.cat([img_emb, txt_emb], dim=0)  
    N, d = X.shape
    print(f"[sae] Training SAE on {N} embeddings of dim {d}")

    code_dim = sae_cfg.get("code_dim", 4096)
    target_k = sae_cfg.get("target_k", 5)
    batch_size = sae_cfg.get("batch_size", 1024)
    lr = float(sae_cfg.get("lr", 5e-4))
    epochs = sae_cfg.get("epochs", 20)
    seed = sae_cfg.get("seed", 0)

    torch.manual_seed(seed)

    sae = BatchTopKSAE(d_in=d, m=code_dim, target_k=target_k).to(DEVICE)

    ds = TensorDataset(X)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS
    )

    opt = torch.optim.AdamW(sae.parameters(), lr=lr)

    sae.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_num = 0
        for (x_batch,) in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}"):
            x_batch = x_batch.to(DEVICE)

            opt.zero_grad()
            x_hat, _, _ = sae(x_batch)
            loss = ((x_batch - x_hat) ** 2).mean()
            loss.backward()
            opt.step()

            sae.normalize_dictionary()

            total_loss += loss.item() * x_batch.size(0)
            total_num += x_batch.size(0)

        avg_loss = total_loss / total_num
        print(f"[sae] Epoch {epoch+1}/{epochs} MSE={avg_loss:.6f}")

    sae_dir = sae_cfg.get("sae_dir", "./outputs/sae")
    os.makedirs(sae_dir, exist_ok=True)
    ckpt_path = os.path.join(sae_dir, "sae_checkpoint.pt")
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "d_in": d,
            "code_dim": code_dim,
            "target_k": target_k,
            "cfg": cfg,
        },
        ckpt_path,
    )
    print(f"[sae] Saved SAE checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train_sae(args.config)
