import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from src.model.gpt2_model import GPTModel_Torch
from src.model.tokenizer import GPT2Tokenizer
from src.data.dataset import GPT2Dataset
import time
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import mlflow
import hydra
from omegaconf import DictConfig
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

def plot_losses(epochs_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 

    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("Hydra output dir:", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)


    # -----------------------------
    # Load dataset (Wikitext)
    # -----------------------------
    ds = load_dataset(cfg.data.dataset_name,cfg.data.dataset_config)
    train_text = " ".join(ds["train"]["text"])
    val_text   = " ".join(ds["validation"]["text"])

    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer = GPT2Tokenizer()
    
    # -----------------------------
    # Datasets / Loaders
    # -----------------------------
    context_length = cfg.data.context_length
    batch_size = cfg.training.batch_size
    num_workers = cfg.data.num_workers
    train_dataset = GPT2Dataset(txt=train_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)
    val_dataset   = GPT2Dataset(txt = val_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size = batch_size, shuffle = True, drop_last = True, num_workers = num_workers)
    

    # -----------------------------
    # Mlflow
    # -----------------------------
    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("GPT2_Implementation")

    run_name = f"gpt2_run_{cfg.model.embed_dim}d_{cfg.training.epochs}epochs"
    

    with mlflow.start_run(run_name=run_name):
        # Log config parameters
        mlflow.log_params(cfg.model)
        mlflow.log_params(cfg.training)
    # -----------------------------
    # Model
    # -----------------------------
        model = GPTModel_Torch(
            vocab_size=cfg.model.vocab_size,
            max_len=cfg.model.max_len,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(model, optimizer, criterion=criterion, device=device, cfg=cfg,log_freq=40)
        trainer.train(train_loader, val_loader)

        epochs_tensor = torch.linspace(0, cfg.training.epochs, len(trainer.train_losses))
        plot_losses(epochs_tensor, trainer.train_losses, trainer.val_losses)


if __name__ == "__main__":
    main()