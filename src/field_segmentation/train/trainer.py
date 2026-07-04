"""Model training orchestration."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from field_segmentation.eval.metrics import Metrics, eval_model


class Trainer:
    def __init__(
        self,
        model_config: dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.config = model_config["training"]
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self) -> None:
        torch.manual_seed(self.config["seed"])  # for reproducibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_history = []
        val_history = []

        best_val_iou = float("-inf")
        best_iteration = -1

        num_epochs = self.config.get("n_epochs", self.config.get("epochs", 1))
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            loop = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")

            for batch in loop:
                images = batch["image"]
                masks = batch["mask"]
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
            if val_metrics.iou > best_val_iou:
                best_val_iou = val_metrics.iou
                best_iteration = epoch
                torch.save(model.state_dict(), "unet_best.pth")

            print(
                f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics.loss:.4f} | Val IoU: {val_metrics.iou:.4f} | "
                f"Val Dice: {val_metrics.dice:.4f}"
            )

        print(
            f"Best model saved at epoch {best_iteration+1} with val iou {best_val_iou:.4f}"
        )
