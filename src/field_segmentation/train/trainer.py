"""Model training orchestration."""

from typing import Any
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.field_segmentation.eval.metrics import eval_model, Metrics


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.config = config["trainer"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        pass

    def train(self):

        torch.manual_seed(self.config["seed"])  # for reproducibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_history = []
        val_history = []

        min_val_iou = float("inf")
        best_iteration = -1

        num_epochs = self.config["num_epochs"]
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")

            for images, masks in loop:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                loop.set_postfix(loss=loss.item())

            train_loss = train_loss / len(self.train_loader)
            train_history.append(train_loss)

            val_metrics : Metrics = eval_model(
                model, criterion, self.val_loader, device=torch.device(device)
            )

            val_history.append(val_metrics)

            # Save best model based on validation loss
            if val_metrics.iou < min_val_iou:
                min_val_iou = val_metrics.iou
                best_iteration = epoch
                torch.save(model.state_dict(), "unet_best.pth")

            print(
                f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics:.4f} | Val IoU: {val_metrics.iou:.4f} | Val Dice: {val_metrics.dice:.4f}"
            )

        print(
            f"Best model saved at epoch {best_iteration+1} with val iou {min_val_iou:.4f}"
        )
