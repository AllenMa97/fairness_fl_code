"""Task-agnostic model adapters shared by the FedFACT / PraFFL baselines.

Both baselines were originally written against the BERT-only interface
(``bert`` / ``drop`` / ``out`` / ``encode``).  The repository exposes the same
"shared backbone + binary head" structure for the other modalities:

    SENT_CLF     BertClassifier  bert / drop / out(2)       forward(input_ids, attention_mask)
    IMG_CLF      RegularCNN      shared_base / out_layer(1) forward(x)
    AUD_CLF      AudioConvRNN    shared_base / out_layer(1) forward(x)
    Tabular_CLF  RegularANN      shared_base / out_layer(1) forward(x)

The helpers below hide that difference so the algorithm code stays identical
across text, image, audio and tabular tasks.
"""

from __future__ import annotations

import torch

TEXT_TASK = "SENT_CLF"
IMAGE_TASK = "IMG_CLF"
AUDIO_TASK = "AUD_CLF"
TABULAR_TASK = "Tabular_CLF"
SUPPORTED_TASKS = (TEXT_TASK, IMAGE_TASK, AUDIO_TASK, TABULAR_TASK)


def normalize_task(task) -> str:
    """Map a task string (possibly with a suffix) to one of the four modalities."""
    task = str(task)
    for candidate in SUPPORTED_TASKS:
        if candidate in task:
            return candidate
    raise ValueError(f"unsupported task {task!r}; expected one of {SUPPORTED_TASKS}")


def is_text_task(task) -> bool:
    return normalize_task(task) == TEXT_TASK


def move_batch_inputs(batch, task, device) -> dict[str, torch.Tensor]:
    """Move only the tensors that the model consumes onto ``device``."""
    task = normalize_task(task)
    if task == TEXT_TASK:
        return {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        }
    key = "img" if task in (IMAGE_TASK, AUDIO_TASK) else "X"
    return {"x": batch[key].to(device, non_blocking=True)}


def encoder_module(model: torch.nn.Module, task) -> torch.nn.Module:
    """Return the communicated backbone (BERT encoder / shared_base)."""
    name = "bert" if is_text_task(task) else "shared_base"
    module = getattr(model, name, None)
    if not isinstance(module, torch.nn.Module):
        raise ValueError(
            f"task {normalize_task(task)} requires a model exposing {name!r}"
        )
    return module


def head_module(model: torch.nn.Module, task) -> torch.nn.Linear:
    """Return the model's own binary classification head."""
    name = "out" if is_text_task(task) else "out_layer"
    head = getattr(model, name, None)
    if not isinstance(head, torch.nn.Linear):
        raise ValueError(
            f"task {normalize_task(task)} requires a linear {name!r} head"
        )
    return head


def feature_dim(model: torch.nn.Module, task) -> int:
    return int(head_module(model, task).in_features)


def _align_dtype(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Match dataset tensors to the module dtype.

    DUTCH ships float64 features while every model is float32; the model
    ``forward`` implementations cast defensively, so the adapter must too when
    it calls the backbone directly.
    """
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return tensor if tensor.dtype == parameter.dtype else tensor.to(parameter.dtype)
    return tensor


def encode_features(model: torch.nn.Module, inputs, task) -> torch.Tensor:
    """Map already-moved model inputs to the representation used by the algorithm."""
    if is_text_task(task):
        dropout = getattr(model, "drop", None)
        features = model.encode(inputs["input_ids"], inputs["attention_mask"])
        return features if dropout is None else dropout(features)
    encoder = encoder_module(model, task)
    return encoder(_align_dtype(encoder, inputs["x"]))


def to_two_class_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalise ``[B,1]`` sigmoid logits to the ``[B,2]`` cross-entropy view."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    if logits.ndim != 2:
        raise ValueError(f"expected rank-2 logits, got shape {tuple(logits.shape)}")
    if logits.shape[1] == 2:
        return logits
    if logits.shape[1] != 1:
        raise ValueError(f"expected one or two logits, got shape {tuple(logits.shape)}")
    # softmax([0, z]) == [1 - sigmoid(z), sigmoid(z)] keeps the binary head's
    # sigmoid probability semantics while reusing the two-class CE/ensemble code.
    return torch.cat((torch.zeros_like(logits), logits), dim=1)


def forward_logits(model: torch.nn.Module, inputs, task) -> torch.Tensor:
    """Return ``[B,2]`` logits for any supported modality."""
    if is_text_task(task):
        result = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"{normalize_task(task)} model must return (features, logits)")
        return to_two_class_logits(result[1])
    features = encode_features(model, inputs, task)
    head = head_module(model, task)
    if features.dtype != head.weight.dtype:
        features = features.to(head.weight.dtype)
    return to_two_class_logits(head(features))
