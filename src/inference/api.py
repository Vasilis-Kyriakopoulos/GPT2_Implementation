import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from src.model.gpt2_model import GPTModel_Torch
from src.model.tokenizer import GPT2Tokenizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler("inference.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

CHAMPION_MODEL_PATH = Path("models/champion/trained_model.pt")
CHAMPION_CONFIG_PATH = Path("models/champion/config.yaml")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(default=40, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    top_k: Optional[int] = Field(default=25, ge=1, le=200)


class GenerateResponse(BaseModel):
    generated_text: str


class ChampionModelService:
    def __init__(self) -> None:
        self.model: Optional[GPTModel_Torch] = None
        self.tokenizer = GPT2Tokenizer()
        self.cfg = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()

    def _pull_champion_from_dvc(self) -> None:
        paths = [str(CHAMPION_MODEL_PATH), str(CHAMPION_CONFIG_PATH)]
        cmd = [sys.executable, "-m", "dvc", "pull", *paths]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("DVC pull failed: %s", result.stderr.strip())
            raise RuntimeError(f"DVC pull failed: {result.stderr.strip()}")
        logger.info("Champion model pulled from DVC.")

    def refresh(self) -> None:
        with self._lock:
            self._pull_champion_from_dvc()
            self.cfg = OmegaConf.load(CHAMPION_CONFIG_PATH)
            model = GPTModel_Torch(
                vocab_size=self.cfg.model.vocab_size,
                max_len=self.cfg.model.max_len,
                embed_dim=self.cfg.model.embed_dim,
                num_layers=self.cfg.model.num_layers,
                num_heads=self.cfg.model.num_heads,
                dropout=self.cfg.model.dropout,
            )
            state_dict = torch.load(CHAMPION_MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Champion model loaded on %s.", self.device)

    def generate(self, payload: GenerateRequest) -> str:
        if self.model is None or self.cfg is None:
            raise RuntimeError("Champion model is not loaded.")

        with self._lock:
            prompt_ids = self.tokenizer.encode(payload.prompt)
            if not prompt_ids:
                raise ValueError("Prompt cannot be empty after tokenization.")

            max_len = int(self.cfg.model.max_len)
            if len(prompt_ids) > max_len:
                prompt_ids = prompt_ids[-max_len:]

            max_allowed = max_len - len(prompt_ids)
            if max_allowed <= 0:
                raise ValueError(
                    f"Prompt is too long for max_len={max_len}; shorten prompt."
                )

            max_new_tokens = min(payload.max_new_tokens, max_allowed)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    context_length=max_len,
                    temperature=payload.temperature,
                    top_k=payload.top_k,
                )
            return self.tokenizer.decode(out_ids[0].tolist())


service = ChampionModelService()
app = FastAPI(title="GPT2 Champion Inference API", version="1.0.0")


@app.on_event("startup")
def startup_event() -> None:
    service.refresh()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if service.model is not None else "not_ready",
        "device": str(service.device),
        "champion_model_path": str(CHAMPION_MODEL_PATH),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_text(payload: GenerateRequest) -> GenerateResponse:
    try:
        text = service.generate(payload)
        return GenerateResponse(generated_text=text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Generation failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reload")
def reload_champion() -> dict:
    try:
        service.refresh()
        return {"status": "reloaded"}
    except Exception as exc:  # pragma: no cover
        logger.exception("Reload failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
