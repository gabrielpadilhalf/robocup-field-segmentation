"""Model training orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from field_segmentation.eval.metrics import Metrics, eval_model

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class Trainer:
    def __init__(
        self,
        model_config: dict[str, Any],
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.model_config = model_config
        self.config = model_config["training"]
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _save_training_plot(
        self,
        train_history: list[float],
        val_history: list[Metrics],
    ) -> Path:
        model_name = self.model_config["model"]["name"]
        plots_dir = Path(self.model_config["paths"]["training_plots_dir"])
        plots_dir.mkdir(parents=True, exist_ok=True)
        output_path = plots_dir / f"{model_name}_training_metrics.png"

        epochs = list(range(1, len(train_history) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_history, label="train loss")
        plt.plot(epochs, [metric.loss for metric in val_history], label="val loss")
        plt.plot(epochs, [metric.iou for metric in val_history], label="val iou")
        plt.plot(epochs, [metric.dice for metric in val_history], label="val dice")
        plt.title(f"Training Metrics - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    def train(self) -> None:
        torch.manual_seed(self.config["seed"])  # for reproducibility
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.model.to(device)
        checkpoints_dir = Path(self.model_config["paths"]["checkpoints_dir"])
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        model_name = self.model_config["model"]["name"]
        best_weights_path = checkpoints_dir / f"{model_name}_best.pth"

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

            val_metrics: Metrics = eval_model(
                model,
                criterion,
                self.val_loader,
                torch.device(device),
                desc=f"Val Epoch {epoch+1}/{num_epochs}",
            )

            val_history.append(val_metrics)

            # Save best model based on validation iou
            if val_metrics.iou > best_val_iou:
                best_val_iou = val_metrics.iou
                best_iteration = epoch
                torch.save(model.state_dict(), best_weights_path)

            print(
                f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics.loss:.4f} | Val IoU: {val_metrics.iou:.4f} | "
                f"Val Dice: {val_metrics.dice:.4f}"
            )

        print(
            f"Best model saved at epoch {best_iteration+1} with val iou "
            f"{best_val_iou:.4f} at {best_weights_path}"
        )
        plot_path = self._save_training_plot(train_history, val_history)
        print(f"Training metrics plot saved to {plot_path}")
