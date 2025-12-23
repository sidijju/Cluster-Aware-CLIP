# Cluster-Aware-CLIP

### Setup for CLIP Base and DEC

- Run `make setup` and run if its a fresh CPU instance
- Activate the venv with `source .venv/bin/activate`
- If you want to activate wandb logging, run 'wandb login'
- Download data based on each dataset with the relevant make command i.e. `make coco`
- Change arguments in `config.yaml`
- Run script with `python3 generate_embeddings.py`

### Setup for SAE

- Use 'config_clip.yaml' settings
- Run 'extract_embeddings_clip.py'
- Run 'train_sae.py'
- Run 'analyze_geometry.py'
- Run 'visualize_concepts.py'

You can find the final report in final_report.pdf
