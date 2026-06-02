"""Training loops for optional V2 ResNet18 classifiers."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm


def train_multilabel_epoch(model, loader, optimizer, device, amp: bool = True, scaler=None) -> float:
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    losses = []
    amp_enabled = bool(amp and device.type == "cuda")
    for images, labels, _ in tqdm(loader, desc="train global", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        _step(loss, optimizer, scaler, amp_enabled)
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(len(losses), 1))


def train_multiclass_epoch(model, loader, optimizer, device, amp: bool = True, scaler=None) -> float:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    amp_enabled = bool(amp and device.type == "cuda")
    for images, labels, _ in tqdm(loader, desc="train crop", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        _step(loss, optimizer, scaler, amp_enabled)
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def evaluate_multilabel_loss(model, loader, device) -> float:
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    losses = []
    for images, labels, _ in tqdm(loader, desc="valid global", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(model(images), labels)
        losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def evaluate_multiclass_loss(model, loader, device) -> float:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0
    for images, labels, _ in tqdm(loader, desc="valid crop", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(float(loss.detach().cpu().item()))
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return float(sum(losses) / max(len(losses), 1)), float(correct / max(total, 1))


def _step(loss, optimizer, scaler, amp_enabled: bool) -> None:
    if scaler is not None and amp_enabled:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
