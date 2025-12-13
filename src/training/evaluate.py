import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.model.gpt2_model import GPTModel_Torch
from src.model.tokenizer import GPT2Tokenizer
from src.data.dataset import GPT2Dataset
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import mlflow
from mlflow.models import infer_signature
import hydra
from omegaconf import DictConfig
from src.training.trainer import Trainer
from urllib.parse import urlparse
import dagshub
from hydra.utils import to_absolute_path


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


os.environ['MLFLOW_TRACKING_URI']="https://github.com/Vasilis-Kyriakopoulos/GPT2_Implementation.git"
os.environ['MLFLOW_TRACKING_USERNAME']="Vasilis-Kyriakopoulos"
os.environ["MLFLOW_TRACKING_PASSWORD"]="c0966aa2cc5d78f4726ec30ecdf6a002565163df"





@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("Hydra output dir:", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # -----------------------------
    # Load dataset (Wikitext)
    # -----------------------------
    ds = load_dataset(cfg.data.dataset_name,cfg.data.dataset_config)
    test_text = " ".join(ds["test"]["text"])

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
    train_dataset = GPT2Dataset(txt=test_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)
   
    test_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, num_workers = num_workers)

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

    run_name = f"gpt2_run_{cfg.model.embed_dim}"

    with mlflow.start_run(run_name=run_name):
      
        # Log config parameters
        mlflow.log_params(cfg.model)
        mlflow.log_params(cfg.training)


        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)
        criterion = nn.CrossEntropyLoss()
        model.load_state_dict(torch.load(to_absolute_path('models/best_model.pt')))
        evaluator = Trainer(model, optimizer, criterion=criterion, device=device, cfg=cfg,log_freq=0)
        test_loss = evaluator.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        mlflow.log_metric("test_loss", test_loss)



if __name__ == "__main__":
    main()