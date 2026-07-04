"""Segmentation metrics such as IoU and Dice."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from tqdm import tqdm


@dataclass(frozen=True)
class Metrics:
    """Metrics for segmentation tasks."""

    iou: float
    dice: float
    loss: float


def eval_model(
    model: nn.Module,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str | None = None,
) -> Metrics:
    """
    Evaluate UNet model on training, validation, and test sets.
    Args:
        model (nn.Module): The UNet model to evaluate.
        criterion (nn.Module): Loss function to compute the loss.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on (CPU or GPU).
    Returns:
        tuple: A tuple containing the average training loss, validation loss, and test loss.

    """
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        iterator = tqdm(loader, desc=desc) if desc is not None else loader
        for batch in iterator:
            images = batch["image"]
            masks = batch["mask"]
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks).item()
            prediction = torch.argmax(outputs, dim=1)
            target = masks
            total_true_positives += ((prediction == 1) & (target == 1)).sum().item()
            total_false_positives += ((prediction == 1) & (target != 1)).sum().item()
            total_false_negatives += ((prediction != 1) & (target == 1)).sum().item()
            if desc is not None:
                iterator.set_postfix(loss=total_loss / max(1, len(loader)))

    EPSILON = 1e-7
    iou = total_true_positives / (
        total_true_positives + total_false_positives + total_false_negatives + EPSILON
    )
    dice = 2 * (
        total_true_positives
        / (
            2 * total_true_positives
            + total_false_positives
            + total_false_negatives
            + EPSILON
        )
    )

    return Metrics(iou=iou, dice=dice, loss=total_loss / len(loader))
