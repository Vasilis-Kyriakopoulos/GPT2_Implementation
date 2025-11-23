import torch
import logging
import mlflow
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

class Trainer:
        def __init__(self, model, optimizer, criterion, device, cfg,log_freq):
                self.model = model
                self.optimizer = optimizer
                self.criterion = criterion
                self.device = device
                self.cfg = cfg
                self.log_freq = log_freq
                self.train_losses = []
                self.val_losses = []


        def evaluate(self,dataloader):
            self.model.eval()
            total_loss = 0

            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    logits = self.model(x)
                    loss = self.criterion( 
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
                    total_loss += loss.item()

            return total_loss / len(dataloader)
        
        def train(self,train_loader,val_loader):
            epochs = self.cfg.training.epochs
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                bad_epochs = 0
                best_val_loss = float("inf")
                for i, (x, y) in enumerate(train_loader):

                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()

                    logits = self.model(x)                    # (B, T, vocab)
                    loss = self.criterion( 
                        logits.view(-1, logits.size(-1)), 
                        y.view(-1)
                    )           

                    loss.backward()
                    self.optimizer.step()

                    if i % self.log_freq == 0:
                        logger.info(f"step:{i+1}/{len(train_loader)}, loss:{loss.item():.4f}")

                    total_loss += loss.item()

                train_loss = total_loss / len(train_loader)
                val_loss = self.evaluate(val_loader)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                logger.info(f"Epoch {epoch+1}/{epochs} - train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
          #     torch.save(model.state_dict(), f"models/gpt2_epoch{epoch+1}.pt")

                # Log metrics
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    bad_epochs = 0
                    torch.save(self.model.state_dict(), "models/best_model.pt")
                    logger.info("New best model saved!")
                    mlflow.log_metric("best_val_loss", best_val_loss)
                    mlflow.pytorch.log_model(self.model, name="best_model")

                else:
                    bad_epochs += 1
                    logger.info(f"No improvement (bad epochs: {bad_epochs})")

                if bad_epochs >= self.cfg.training.patience:
                    logger.info("EARLY STOPPING TRIGGERED.")
                    break

                time.sleep(4)
            
