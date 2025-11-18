import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.model.gpt2_model import GPTModel_Torch
from src.model.tokenizer import GPT2Tokenizer
from src.data.dataset import GPT2Dataset


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        print(x.shape)
        optimizer.zero_grad()

        logits = model(x)                    # (B, T, vocab)
        loss = nn.CrossEntropyLoss()( 
            logits.view(-1, logits.size(-1)), 
            y.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = nn.CrossEntropyLoss()( 
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # -----------------------------
    # Load dataset (Wikitext)
    # -----------------------------
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = " ".join(ds["train"]["text"])
    val_text   = " ".join(ds["validation"]["text"])

    # -----------------------------
    # Tokenizer (trained earlier)
    # -----------------------------
    tokenizer = GPT2Tokenizer()
    tokenizer.train(train_text)
    # -----------------------------
    # Datasets / Loaders
    # -----------------------------
    block_size = 128
    batch_size = 32

    train_dataset = GPT2Dataset(train_text, tokenizer, block_size)
    val_dataset   = GPT2Dataset(val_text,   tokenizer, block_size)
    print(f"Train dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    # -----------------------------
    # Model
    # -----------------------------
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    model = GPTModel_Torch(
        vocab_size=vocab_size,
        max_len=block_size,
        embed_dim=256,
        num_layers=4,
        num_heads=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # -----------------------------
    # Training Loop
    # -----------------------------
    epochs = 5

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

        # Save checkpoint at each epoch
        torch.save(model.state_dict(), f"models/gpt2_epoch{epoch+1}.pt")


if __name__ == "__main__":
    main()
