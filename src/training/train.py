import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from src.model.gpt2_model import GPTModel_Torch
from src.model.tokenizer import GPT2Tokenizer
from src.data.dataset import GPT2Dataset
import time


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(dataloader):
        if i%20 == 0:
            print(f"steps:{i+1}/{len(dataloader)}")
        x = x.to(device)
        y = y.to(device)
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

    
    tokenizer = GPT2Tokenizer(
    "tokenizers/gpt_tokenizer/vocab.json",
    "tokenizers/gpt_tokenizer/merges.txt"
    )
    #tokenizer.train(train_text,vocab_size=30000,save_dir='tokenizers/gpt_tokenizer')
    
    # -----------------------------
    # Datasets / Loaders
    # -----------------------------
    context_length = 512
    batch_size = 32

    train_dataset = GPT2Dataset(txt=train_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)
    val_dataset   = GPT2Dataset(txt = val_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)
    print(f"Train dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,drop_last = True)

    # -----------------------------
    # Model
    # -----------------------------
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    model = GPTModel_Torch(
        vocab_size=vocab_size,
        max_len=context_length,
        embed_dim=1024,
        num_layers=4,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1
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
        time.sleep(4)
        # Save checkpoint at each epoch
        torch.save(model.state_dict(), f"models/gpt2_epoch{epoch+1}.pt")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)
    main()
    tokenizer = GPT2Tokenizer(
    "tokenizers/gpt_tokenizer/vocab.json",
    "tokenizers/gpt_tokenizer/merges.txt"
    )
    vocab_size = tokenizer.tokenizer.get_vocab_size()

    model = GPTModel_Torch(
        vocab_size=vocab_size,
        max_len=512,
        embed_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
    torch.load("models/gpt2_epoch3.pt", map_location=device)
    )

    prompt_ids = tokenizer.encode("The war ")
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    model.to(device)

    generated_ids = model.generate(prompt_ids)
    print(tokenizer.decode(generated_ids[0].tolist()))


