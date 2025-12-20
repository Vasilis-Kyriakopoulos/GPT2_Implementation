import json
import shutil
import os

CAND_DIR = "models/candidate"
CHAMP_DIR = "models/champion"

cand_metrics = json.load(open(f"{CAND_DIR}/metrics.json"))

# Case 1: no champion yet → promote immediately
if not os.path.exists(CHAMP_DIR):
    shutil.copytree(CAND_DIR, CHAMP_DIR)
    print("No champion found. Promoted candidate as first champion.")
    exit(0)

# Case 2: champion exists → compare
champ_metrics = json.load(open(f"{CHAMP_DIR}/metrics.json"))

if cand_metrics["val_loss"] < champ_metrics["val_loss"]:
    shutil.rmtree(CHAMP_DIR)
    shutil.copytree(CAND_DIR, CHAMP_DIR)
    print("Promoted candidate to champion")
else:
    print("Candidate worse, no promotion")
