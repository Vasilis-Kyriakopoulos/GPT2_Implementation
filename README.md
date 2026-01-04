# GPT2_Implementation (refactor_mlops)

Implementation of a GPT-2 style language model with an MLOps-focused refactor.  
The repository emphasizes reproducibility, experiment tracking, automated pipelines, and containerized execution rather than model performance.

---

## Overview

This project contains:
- A GPT-2–like language model implementation (decoder-only Transformer)
- An end-to-end MLOps setup for training, evaluation, and model promotion
- Reproducible pipelines and automated evaluation

The goal is to demonstrate **practical MLOps concepts** such as experiment tracking, data and model versioning, CI-based evaluation, and containerized execution.

---

## Repository Structure
```bash
.github/workflows/ # CI workflows (tests and evaluation)
configs/ # YAML configuration files
data/ # Datasets (tracked with DVC)
models/
├── candidate/ # Latest trained candidate model
└── champion/ # Promoted (best) model
src/
├── data/ # Data preprocessing scripts
├── model/ # Model classes
└── training/ # Training, evaluation, generation, promotion
tests/ # Automated tests
dvc.yaml # DVC pipeline definition
dvc.lock # Locked pipeline versions
DockerFile # Docker image definition (CPU PyTorch)
docker-compose.yml # Local container execution
requirements.txt # dependencies 
requirements_cpu.txt # CPU dependencies pytorch
requirements_gpu.txt # GPU dependencies pytorch (optional)
```

---

## MLOps Components

### Experiment Tracking
- **MLflow** is used to log parameters, metrics, artifacts, and trained models.
- Runs are stored remotely using **DAGsHub**, making experiments accessible and reproducible without relying on local storage.

### Data & Model Versioning
- **DVC** is used to track datasets and model artifacts.
- All major steps are defined as pipeline stages, enabling reproducible execution.

### Model Lifecycle
- Models are first trained as **candidate** models.
- After evaluation, the candidate can be promoted to **champion** if it performs better.
- The champion model represents the best available version and is the one evaluated in CI.

### Containerization
- **Docker** provides a consistent runtime environment.
- The container uses **CPU PyTorch** to keep images lightweight and CI-friendly.

### Continuous Integration
- **GitHub Actions** automatically run tests and evaluate the champion model inside a Docker container on every commit.

---

## DVC Pipeline

The pipeline consists of the following stages:
- `training_data_prep`: preprocess and split the dataset (train/val/test)
- `upload_data`: push prepared data to remote storage
- `training`: train the model and produce a candidate model
- `evaluate`: evaluate the candidate model and compute metrics
- `upload_model`: push model artifacts and metrics
- `promote_model`: promote candidate to champion based on metrics

Run the full pipeline:
```bash
dvc repro
``` 
### Running Locally
Setup environment
```bash
python -m venv .venv
```


source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows
```bash
pip install -r requirements.txt
pip install -r requirements_cpu.txt
```
# Train the model
```bash
python -m src.training.train
```
# Generate text
```bash
python -m src.training.generate 
```

# MLflow + DAGsHub Setup (Example)
```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
export MLFLOW_TRACKING_USERNAME="<username>"
export MLFLOW_TRACKING_PASSWORD="<token>"
```

# Then run training normally:
```bash
python -m src.training.train
```
# Docker Usage
Build and run the container:
```bash
docker build -t gpt2-mlops -f DockerFile .
docker run --rm -it gpt2-mlops
```


Notes

The focus of this project is MLOps practices, not achieving state-of-the-art model performance.

GPU support can be added by switching to CUDA-based Docker images and GPU dependencies.

License
