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
import wandb

NUM_WORKERS = os.cpu_count()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEBUG = False

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
    processor = CLIPProcessor.from_pretrained(config['model']['model_name'], use_fast=True)

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

    # write df to parquet
    print("Saving embeddings...")
    df = pd.DataFrame(all_rows)
    table = pa.Table.from_pandas(df)
    output_dir = config['output'].get('output_dir','./clip_cpu_model')
    path = output_dir + '/embeddings.parquet'
    pq.write_table(table, path)
    print(f"Saved embeddings to {path}")

def save_clusters(config):
    output_dir = config['output'].get('output_dir','./clip_cpu_model')
    embedding_path = output_dir + '/embeddings.parquet'
    centers_path = output_dir + '/dec_centers.pt'
    clusters_path = output_dir + '/clusters.parquet'
    
    df = pd.read_parquet(embedding_path)
    X = np.stack(df['embedding'].values)
    
    centers = torch.load(centers_path)
    centers = torch.tensor(centers).float()

    X_t = torch.tensor(X)

    dist = ((X_t.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
    q = (1.0 + dist).pow(-1)
    q = q / q.sum(dim=1, keepdim=True)

    clusters = q.argmax(dim=1).numpy()

    df_out = pd.DataFrame({
        "label": df["label"],
        "cluster": clusters,
        "q": list(q.numpy()),
    })
    df_out.to_parquet(out_file)

    print(f"[DEC] Clusters saved to {clusters_path}")

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

def compute_alignment_accuracy(logits, labels): 
    pred_i2t = logits.argmax(dim=1)
    pred_t2i = logits.t().argmax(dim=1)

    acc_i2t = (pred_i2t == labels).float().mean()
    acc_t2i = (pred_t2i == labels).float().mean()

    return 0.5 * (acc_i2t + acc_t2i)

def compute_dec_loss(img_emb, text_emb, centers, epsilon=1e-8):
    # distances to cluster centers
    dist_img = ((img_emb.unsqueeze(1) - centers.unsqueeze(0))**2).sum(dim=2)  # (B, K)
    dist_txt = ((text_emb.unsqueeze(1) - centers.unsqueeze(0))**2).sum(dim=2) # (B, K)

    # student-t kernel soft assignments
    q_img = (1.0 + dist_img).pow(-1)
    q_img = q_img / q_img.sum(dim=1, keepdim=True)

    q_text = (1.0 + dist_txt).pow(-1)
    q_text = q_txt / q_txt.sum(dim=1, keepdim=True)

    # combine image + text q
    q_combined = torch.cat([q_img, q_txt], dim=0)
    f_k = q_combined.sum(dim=0)

    # target distribution p
    p_combined = (q_combined**2) / (f_k + epsilon)
    p_combined = p_combined / p_combined.sum(dim=1, keepdim=True)

    # split back for image and text
    B = img_emb.size(0)
    p_img = p_combined[:B]
    p_txt = p_combined[B:]

    # KL loss
    loss_dec_img = (p_img * (torch.log(p_img + epsilon) - torch.log(q_img + epsilon))).sum(dim=1).mean()
    loss_dec_txt = (p_txt * (torch.log(p_txt + epsilon) - torch.log(q_txt + epsilon))).sum(dim=1).mean()
    return loss_dec_img + loss_dec_txt

def train(config):
    # Wandb logging
    wandb_enabled = config.get("wandb").get("enabled", False)
    dec_enabled = config['dec'].get('enabled', False)
    if wandb_enabled:
        wandb.init(
            project=config["wandb"].get("project", "clip-unsupervised"),
            config=config,
            name="debug" if DEBUG else config["wandb"].get("run_name", "trial")
        )

    dataset = load_dataset(config)
    loader = DataLoader(
        dataset, 
        batch_size=config['training'].get('batch_size', 4), 
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    
    # load openai clip 
    model_name = config['model'].get('model_name','openai/clip-vit-base-patch16')
    model = CLIPModel.from_pretrained(model_name).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['training'].get('lr', 1e-5)))

    if dec_enabled:
        # DEC configurations
        K = config['dec'].get("clusters", 80)
        lambda_dec = config['dec'].get("lambda", 0.1)

        # learnable cluster centers
        centers = nn.Parameter(torch.randn(K, 512, device=DEVICE)) # 512 = default CLIP embedding dimension
        optimizer.add_param_group({"params": centers, "lr": float(config['dec'].get('lr', 1e-4))})

    all_losses = []

    epochs = config['training'].get('epochs', 1)
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

            # DEC cluster computations
            if dec_enabled:
                loss_dec = compute_dec_loss(img_emb, text_emb, centers)

            loss_clip = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2
            if dec_enabled:
                loss = loss_clip + lambda_dec * loss_dec
            else:
                loss = loss_clip

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            clip_alignment_acc = compute_alignment_accuracy(logits, labels)
            if wandb_enabled:
                wandb.log({
                    "loss_total": loss.item(),
                    "loss_clip": loss_clip.item(),
                    "clip_alignment_acc": clip_alignment_acc,
                })

                if dec_enabled:
                    wandb.log({
                        "loss_dec": loss_dec.item(),
                    })

            if DEBUG:
                break

        print(f"Epoch {epoch+1} loss={loss.item():.4f}")
        if wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": loss.item()
            })

    out_dir = config['output'].get('output_dir','./clip_cpu_model')
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    plot_losses(all_losses)

    save_embeddings(model, dataset, config)
    if dec_enabled:
        torch.save(centers.detach().cpu(), os.path.join(out_dir, "dec_centers.pt"))
        save_clusters(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
