
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


def show_top_examples(concept_id: int, codes: torch.Tensor, captions, k: int = 10):
    activations = codes[:, concept_id].numpy()  # [N]
    idxs = np.argsort(-activations)[:k]
    print(f"\n=== Concept {concept_id} top {k} examples ===")
    for i in idxs:
        print(f"({activations[i]:.4f}) {captions[i]}")


def main(config_path: str, concept_id: int, k: int):
    cfg = load_config(config_path)

    img_emb, txt_emb, captions = load_embeddings(cfg)
    sae, d_in, code_dim, target_k = load_sae(cfg)

    if concept_id < 0 or concept_id >= code_dim:
        raise ValueError(f"concept_id must be in [0, {code_dim-1}]")

    print(f"[viz] Using concept {concept_id} out of {code_dim}")

    # Compute codes
    img_codes = encode_all_codes(sae, img_emb)
    txt_codes = encode_all_codes(sae, txt_emb)

    # Show top examples for images and texts separately
    print("\n[Images / captions as proxy]:")
    show_top_examples(concept_id, img_codes, captions, k=k)

    print("\n[Texts only]:")
    show_top_examples(concept_id, txt_codes, captions, k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--concept_id", type=int, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    main(args.config, args.concept_id, args.k)
