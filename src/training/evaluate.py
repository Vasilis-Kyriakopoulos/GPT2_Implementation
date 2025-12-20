import json
import os

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.model.gpt2_model import GPTModel_Torch
from src.model.tokenizer import GPT2Tokenizer
from src.data.dataset import GPT2Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import mlflow
from omegaconf import DictConfig
from omegaconf import OmegaConf
from src.training.trainer import Trainer
from urllib.parse import urlparse
import dagshub
from hydra.utils import to_absolute_path
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model weights (.pt)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/metrics.json",
        help="Where to write metrics JSON"
    )
    return parser.parse_args()


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME","")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD","")


def main():
    args = parse_args()
    logger.info(f"Model path:{args.model_path}")
    logger.info(f"Output path:{args.output_path}")
    logger.info(f"Config path:{args.config_path}")
    model_path = args.model_path
    output_path = args.output_path
    config_path = args.config_path    
    cfg = OmegaConf.load(config_path)

    # -----------------------------
    # Load dataset (Wikitext)
    # -----------------------------
    test_dataset_name = cfg.data.test_dataset_name    
    test_text = ""

    
    try:
        with open(to_absolute_path(f"data/{test_dataset_name}"), 'r', encoding='utf-8') as f:
            test_text = f.read()
             
    except Exception as e:
        logger.error(f"Error reading files: {e}")
        assert 1 == 1
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
    test_dataset = GPT2Dataset(txt=test_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)
   
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = num_workers)

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
    # -----------------------------
    # Mlflow
    # -----------------------------
    mlflow.set_tracking_uri("https://dagshub.com/Vasilis-Kyriakopoulos/GPT2_Implementation.mlflow")


    mlflow.set_experiment("GPT2_Test_Best_Model")

    run_name = f"gpt2_test_{cfg.model.embed_dim}"

    with mlflow.start_run(run_name=run_name):
      
        # Log config parameters
        mlflow.log_params(cfg.model)
        mlflow.log_params(cfg.training)


        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
        criterion = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(to_absolute_path(model_path), map_location=device))
        evaluator = Trainer(model, optimizer, criterion=criterion, device=device, cfg=cfg,log_freq=0)
        test_loss = evaluator.evaluate(test_loader)
        logger.info(f"Test Loss: {test_loss:.4f}")
        mlflow.log_metric("test_loss", test_loss)
        with open(to_absolute_path(output_path), "w") as f:
            data = {
                "test_loss": test_loss
            }
            json.dump(data, f)

if __name__ == "__main__":
    main()