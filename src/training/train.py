import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from src.model.gpt2_model import GPTModel_Torch
from src.model.tokenizer import GPT2Tokenizer
from src.data.dataset import GPT2Dataset
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_losses(epochs_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()

def train_one_epoch(model, dataloader, optimizer, device,freq_print):
    model.train()
    total_loss = 0
    
    for i, (x, y) in enumerate(dataloader):
        if i%freq_print == 0:
            print(f"steps:{i+1}/{len(dataloader)}")
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        torch.cuda.empty_cache()

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

    
    tokenizer = GPT2Tokenizer()
    
    # -----------------------------
    # Datasets / Loaders
    # -----------------------------
    context_length = 256
    batch_size = 16

    train_dataset = GPT2Dataset(txt=train_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)
    val_dataset   = GPT2Dataset(txt = val_text, tokenizer=tokenizer, stride = context_length,max_length=context_length)
    print(f"Train dataset size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last = True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,drop_last = True)

    # -----------------------------
    # Model
    # -----------------------------
    vocab_size = tokenizer.vocab_size
    model = GPTModel_Torch(
        vocab_size=vocab_size,
        max_len=256,
        embed_dim=256,
        num_layers=6,
        num_heads=4,
        dropout=0.1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # -----------------------------
    # Training Loop
    # -----------------------------
    best_val_loss = float("inf")
    patience = 2            # stop after 2 bad epochs
    bad_epochs = 0
    epochs = 10
    train_losses, val_losses = [],[]
    for epoch in range(epochs):
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device,freq_print=40)
        val_loss = evaluate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        torch.save(model.state_dict(), f"models/gpt2_epoch{epoch+1}.pt")
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            print("New best model saved!")
        else:
            bad_epochs += 1
            print(f"No improvement (bad epochs: {bad_epochs})")

        if bad_epochs >= patience:
            print("EARLY STOPPING TRIGGERED.")
            break

        time.sleep(4)

    epochs_tensor = torch.linspace(0, epochs, len(train_losses))
    plot_losses(epochs_tensor, train_losses, val_losses)



if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)
    #main()
    tokenizer = GPT2Tokenizer()
    vocab_size = tokenizer.vocab_size
    print(vocab_size)
    model = GPTModel_Torch(
        vocab_size=vocab_size,
        max_len=256,
        embed_dim=256,
        num_layers=6,
        num_heads=4,
        dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
    torch.load("./models/best_model.pt", map_location=device)
    )

    prompt_ids = tokenizer.encode("The war started")
    prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    model.to(device)

    generated_ids = model.generate(prompt_ids,max_new_tokens=100,context_length=256,temperature=1,top_k=25)
    print(tokenizer.decode(generated_ids[0].tolist()))
