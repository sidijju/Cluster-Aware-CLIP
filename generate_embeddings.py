import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
import json
import os
import argparse
import matplotlib.pyplot as plt
import tqdm

NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#### Helpers ####

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def normalize_vector(vec):
    return vec / vec.norm(dim=-1, keepdim=True)

def batch_to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}

#### Data ####

def collate_fn(batch):
    imgs, captions = zip(*batch)
    return list(imgs), list(captions)

class CocoDataset(Dataset):
    def __init__(self, root, ann_file):
        self.root = root

        with open(ann_file) as f:
            data = json.load(f)

        self.img_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        self.captions = []
        for annotation in data['annotations']:
            img_file = self.img_id_to_file[annotation['image_id']]
            caption = annotation['caption']
            self.captions.append((img_file, caption))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_file, caption = self.captions[idx]
        img = Image.open(f"{self.root}/{img_file}").convert("RGB")
        return img, caption

# TODO: other datasets

def load_dataset(config):
    name = config['dataset']['name'].lower()

    if name == 'coco':
        return CocoDataset(root=config['dataset']['root'], 
                           ann_file=config['dataset']['ann_file'])
    else:
        raise ValueError(f"Dataset {name} not supported")

def save_embeddings(model, dataset, config):
    model.eval()

    batch_size = config['training']['batch_size']
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    processor = CLIPProcessor.from_pretrained(config['model']['model_name'])

    # get embeddings
    all_rows = []
    print(f"Generating {len(dataset)} embeddings...")
    with torch.no_grad():
        for imgs, texts in tqdm.tqdm(loader):
            inputs = processor(
                images=list(imgs),
                text=list(texts),
                return_tensors="pt",
                padding=True
            )
            inputs = batch_to_device(inputs)

            img_emb = model.get_image_features(pixel_values=inputs['pixel_values'])
            text_emb = model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            imgs_out = normalize_vector(img_emb)
            texts_out = normalize_vector(text_emb)

            for i in range(len(texts)):
                all_rows.append({
                    "label": texts[i],
                    "image_embedding": imgs_out[i].cpu().numpy(),
                    "text_embedding": texts_out[i].cpu().numpy()
                })

            break

    # write df to parquet
    print("Saving embeddings...")
    df = pd.DataFrame(all_rows)
    table = pa.Table.from_pandas(df)
    output_dir = config['output'].get('output_dir','./clip_cpu_model')
    path = output_dir + '/embeddings.parquet'
    pq.write_table(table, path)
    print(f"Saved embeddings to {path}")

#### Train Loop ####

def plot_losses(losses):
    plt.figure(figsize=(8,5))
    plt.plot(losses, label="Batch Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    output_dir = config['output'].get('output_dir','./clip_cpu_model')
    plot_path = output_dir +  '/training_loss.png'
    plt.savefig(plot_path)
    print(f"Saved training loss plot to {plot_path}")
    plt.close()

def train(config):
    dataset = load_dataset(config)
    loader = DataLoader(
        dataset, 
        batch_size=config['training'].get('batch_size', 4), 
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    processor = CLIPProcessor.from_pretrained(config['model']['model_name'])

    # load openai clip 
    model_name = config['model'].get('model_name','openai/clip-vit-base-patch16')
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training'].get('lr', 1e-5)))

    epochs = config['training'].get('epochs', 1)

    all_losses = []
    i=5

    for epoch in range(epochs):
        for imgs, texts in tqdm.tqdm(loader):
            inputs = processor(
                images=list(imgs),
                text=list(texts),
                return_tensors="pt",
                padding=True
            )
            inputs = batch_to_device(inputs)

            img_emb = model.get_image_features(pixel_values=inputs['pixel_values'])
            text_emb = model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            img_emb = normalize_vector(img_emb)
            text_emb = normalize_vector(text_emb)

            logits = img_emb @ text_emb.t()
            labels = torch.arange(len(imgs))
            loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())

            # TODO: remove
            if i == 0:
                break
            else:
                i -= 1

        print(f"Epoch {epoch+1} loss={loss.item():.4f}")

    out_dir = config['output'].get('output_dir','./clip_cpu_model')
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    plot_losses(all_losses)

    save_embeddings(model, dataset, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
