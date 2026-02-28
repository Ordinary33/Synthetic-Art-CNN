from src.logger_config import setup_logger
from src.loaders import get_loaders
from src.model import AIDCNN

from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.scheduler import ReduceLROnPlateau
from tqdm import tqdm

import wandb

wandb.login()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logger(name="train")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    "learning_rate": 0.001,
    "epochs": 15,
    "batch_size": 32,
    "architecture": "Custom CNN",
    "optimizer": "Adam",
}


def train():
    with wandb.init(project="AI-Detection", config=hyperparameters) as run:
        train_loader, val_loader = get_loaders(run.config.batch_size)

        model = AIDCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=run.config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        run.watch(model, criterion, log="all", log_freq=10)

        checkpoint_path = MODELS_DIR / "aid_cnn_model.pth"

        start_epoch = 0
        epoch_no_improve = 0
        early_stopping_patience = 3

        best_val_loss = float("inf")

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "loss" in checkpoint:
                best_val_loss = checkpoint["loss"]
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resumed training from epoch {start_epoch}")

        else:
            logger.info("No Checkpoint found, starting training from scratch")

        for epoch in range(start_epoch, run.config.epochs):
            model.train()
            running_train_loss = 0
            running_train_correct = 0

            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/ {run.config.epochs}",
                leave=False,
            )
            for images, labels in train_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                running_train_correct += torch.sum(predictions == labels).item()
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * images.size(0)
                train_bar.set_postfix(loss=loss.item())

            avg_train_loss = running_train_loss / len(train_loader.dataset)
            train_accuracy = running_train_correct / len(train_loader.dataset)
            logger.info(
                f"Epoch [{epoch + 1} / {run.config.epochs}], Train Loss: {avg_train_loss:.4f}"
            )

            model.eval()
            running_val_loss = 0
            running_val_correct = 0

            val_bar = tqdm(
                val_loader,
                desc=f"Validation Epoch {epoch + 1} / {run.config.epochs}",
                leave=False,
            )

            with torch.no_grad():
                for images, labels in val_bar:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    predictions = torch.argmax(outputs, dim=1)
                    running_val_correct += torch.sum(predictions == labels).item()

                    running_val_loss += loss.item() * images.size(0)
                    val_bar.set_postfix(loss=loss.item())

            avg_val_loss = running_val_loss / len(val_loader.dataset)
            val_accuracy = running_val_correct / len(val_loader.dataset)
            logger.info(
                f"Epoch [{epoch + 1} / {run.config.epochs}], Val Loss: {avg_val_loss:.4f}"
            )
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train_accuracy": train_accuracy,
                    "validation_accuracy": val_accuracy,
                }
            )
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epoch_no_improve = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": best_val_loss,
                    },
                    checkpoint_path,
                )
                logger.info(
                    f"New best model found. Checkpoint saved at epoch {epoch + 1}"
                )

            else:
                epoch_no_improve += 1

                if epoch_no_improve >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break

        logger.info("Training Complete")

        dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE)

        torch.onnx.export(model, dummy_input, "aidcnn.onnx")
        wandb.save("aidcnn.onnx")


if __name__ == "__main__":
    train()
