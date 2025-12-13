from datasets import load_dataset
import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print("Hydra output dir:", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # -----------------------------
    # Load dataset (Wikitext)
    # -----------------------------
    ds = load_dataset(cfg.data.dataset_name,cfg.data.dataset_config)
    train_text = " ".join(ds["train"]["text"])
    val_text   = " ".join(ds["validation"]["text"])
    test_text  = " ".join(ds["test"]["text"])

    os.makedirs("data", exist_ok=True)
    # Save preprocessed data to files
    with open("data/wikitext-train.txt", "w") as f:
        f.write(train_text)
    with open("data/wikitext-val.txt", "w") as f:
        f.write(val_text)
    with open("data/wikitext-test.txt", "w") as f:
        f.write(test_text)

if __name__ == "__main__":
    main()