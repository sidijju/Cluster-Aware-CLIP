
import os
import json
import argparse

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)


def collate_fn(batch):
    imgs, captions = zip(*batch)
    return list(imgs), list(captions)


class CocoDataset(Dataset):
    def __init__(self, root, ann_file):
        self.root = root

        with open(ann_file) as f:
            data = json.load(f)

        self.img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}

        self.captions = []
        for annotation in data["annotations"]:
            img_file = self.img_id_to_file[annotation["image_id"]]
            caption = annotation["caption"]
            self.captions.append((img_file, caption))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_file, caption = self.captions[idx]
        img = Image.open(os.path.join(self.root, img_file)).convert("RGB")
        return img, caption


def load_dataset(cfg):
    name = cfg["dataset"]["name"].lower()
    if name == "coco":
        return CocoDataset(
            root=cfg["dataset"]["root"], ann_file=cfg["dataset"]["ann_file"]
        )
    else:
        raise ValueError(f"Dataset {name} not supported yet")


def extract_embeddings(config_path: str):
    cfg = load_config(config_path)


    dataset = load_dataset(cfg)
    batch_size = cfg["extract"].get("batch_size", 256)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    #frozen clip
    model_name = cfg["model"].get("model_name", "openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    model.eval()

    all_rows = []

    print(f"[extract] Dataset size: {len(dataset)} examples")
    with torch.no_grad():
        for imgs, texts in tqdm.tqdm(loader, desc="Extracting embeddings"):
            inputs = processor(
                images=list(imgs),
                text=list(texts),
                return_tensors="pt",
                padding=True,
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            img_emb = model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_emb = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            img_emb = normalize_vector(img_emb)
            txt_emb = normalize_vector(txt_emb)

            img_emb_np = img_emb.cpu().numpy()
            txt_emb_np = txt_emb.cpu().numpy()

            for i in range(len(texts)):
                all_rows.append(
                    {
                        "caption": texts[i],
                        "image_embedding": img_emb_np[i],
                        "text_embedding": txt_emb_np[i],
                    }
                )

    df = pd.DataFrame(all_rows)
    output_dir = cfg["extract"].get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    emb_filename = cfg["extract"].get("embeddings_filename", "embeddings.parquet")
    path = os.path.join(output_dir, emb_filename)

    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    print(f"[extract] Saved embeddings to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    extract_embeddings(args.config)
