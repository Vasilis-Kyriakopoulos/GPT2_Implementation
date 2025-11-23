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

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)



@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    tokenizer = GPT2Tokenizer()
    model = GPTModel_Torch(
            vocab_size=cfg.model.vocab_size,
            max_len=cfg.model.max_len,
            embed_dim=cfg.model.embed_dim,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout
    
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
    torch.load("./models/best_model.pt", map_location=device)
    )

    prompt_ids = tokenizer.encode("The war started")
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    model.to(device)

    generated_ids = model.generate(prompt_ids,max_new_tokens=100,context_length=256,temperature=1,top_k=25)
    logger.info(f'Generated Text: {tokenizer.decode(generated_ids[0].tolist())}')


if __name__ == "__main__":
    main()
    
