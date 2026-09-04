"""Paper-faithful mathematical building blocks for PraFFL.

PraFFL: A Preference-Aware Scheme in Fair Federated Learning
https://arxiv.org/abs/2404.08973
https://github.com/rG223/PraFFL
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


PRAFFL_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PraFFLConfig:
    tau_c: int
    tau_p: int
    preference_batch_size: int
    hypernetwork_hidden_dim: int
    encoder_learning_rate: float
    hypernetwork_learning_rate: float
    optimizer_name: str
    smooth_gamma: float
    report_preference: tuple[float, float]
    preference_points: int
    preference_chunk_size: int
    hypernetwork_seed_offset: int

    @classmethod
    def from_param_dict(
        cls, param_dict: Mapping[str, object], algorithm_epoch_T: int
    ) -> "PraFFLConfig":
        if algorithm_epoch_T < 2:
            raise ValueError("PraFFL algorithm_epoch_T must be at least 2")

        raw_tau_c = param_dict.get("praffl_tau_c")
        raw_tau_p = param_dict.get("praffl_tau_p")
        if raw_tau_c is None and raw_tau_p is None:
            tau_c = algorithm_epoch_T // 2
            tau_p = algorithm_epoch_T - tau_c
        elif raw_tau_c is None:
            tau_p = int(raw_tau_p)
            tau_c = algorithm_epoch_T - tau_p
        elif raw_tau_p is None:
            tau_c = int(raw_tau_c)
            tau_p = algorithm_epoch_T - tau_c
        else:
            tau_c = int(raw_tau_c)
            tau_p = int(raw_tau_p)
        if tau_c < 1 or tau_p < 1:
            raise ValueError("PraFFL tau_c and tau_p must each be at least 1")
        if tau_c + tau_p != algorithm_epoch_T:
            raise ValueError("PraFFL tau_c + tau_p must equal algorithm_epoch_T")

        raw_report = param_dict.get("praffl_report_preference", (0.5, 0.5))
        report = tuple(float(value) for value in raw_report)
        if (
            len(report) != 2
            or min(report) < 0.0
            or not math.isclose(sum(report), 1.0)
        ):
            raise ValueError(
                "praffl_report_preference must contain two non-negative values summing to 1"
            )

        config = cls(
            tau_c=tau_c,
            tau_p=tau_p,
            preference_batch_size=int(
                param_dict.get("praffl_preference_batch_size", 8)
            ),
            hypernetwork_hidden_dim=int(
                param_dict.get("praffl_hypernetwork_hidden_dim", 256)
            ),
            encoder_learning_rate=float(param_dict.get("learning_rate", 5e-5)),
            hypernetwork_learning_rate=float(
                param_dict.get("praffl_hypernetwork_learning_rate", 1e-3)
            ),
            optimizer_name=str(param_dict.get("optimize_method", "adam")).lower(),
            smooth_gamma=float(param_dict.get("praffl_smooth_gamma", 1.0)),
            report_preference=(report[0], report[1]),
            preference_points=int(param_dict.get("praffl_preference_points", 1000)),
            preference_chunk_size=int(
                param_dict.get("praffl_preference_chunk_size", 128)
            ),
            hypernetwork_seed_offset=int(
                param_dict.get("praffl_hypernetwork_seed_offset", 1701)
            ),
        )
        positive_fields = {
            "praffl_preference_batch_size": config.preference_batch_size,
            "praffl_hypernetwork_hidden_dim": config.hypernetwork_hidden_dim,
            "learning_rate": config.encoder_learning_rate,
            "praffl_hypernetwork_learning_rate": config.hypernetwork_learning_rate,
            "praffl_smooth_gamma": config.smooth_gamma,
            "praffl_preference_points": config.preference_points,
            "praffl_preference_chunk_size": config.preference_chunk_size,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if config.preference_points < 2:
            raise ValueError("praffl_preference_points must be at least 2")
        if config.optimizer_name not in {"adam", "sgd"}:
            raise ValueError("PraFFL optimize_method must be 'adam' or 'sgd'")
        return config


class HyperNetwork(nn.Module):
    """Map a two-objective preference to a binary linear classifier head."""

    def __init__(
        self, preference_dim: int, feature_dim: int, num_classes: int, hidden_dim: int
    ):
        super().__init__()
        if preference_dim != 2:
            raise ValueError("PraFFL requires a two-dimensional preference")
        self.preference_dim = preference_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        output_dim = num_classes * feature_dim + num_classes
        self.network = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, preferences: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if preferences.ndim != 2 or preferences.shape[1] != self.preference_dim:
            raise ValueError("preferences must have shape [num_preferences, 2]")
        generated = self.network(preferences)
        split = self.num_classes * self.feature_dim
        weight = generated[:, :split].reshape(
            -1, self.num_classes, self.feature_dim
        )
        bias = generated[:, split:].reshape(-1, self.num_classes)
        return weight, bias


def functional_linear_heads(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply generated heads without copying tensors or breaking autograd."""
    if features.ndim != 2 or weight.ndim != 3 or bias.ndim != 2:
        raise ValueError("features, weight, and bias must have ranks 2, 3, and 2")
    if weight.shape[0] != bias.shape[0]:
        raise ValueError("weight and bias preference counts must match")
    if weight.shape[1] != bias.shape[1] or weight.shape[2] != features.shape[1]:
        raise ValueError("generated head dimensions do not match features")
    return torch.stack(
        [
            F.linear(features, head_weight, head_bias)
            for head_weight, head_bias in zip(weight, bias)
        ],
        dim=0,
    )


def preference_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return one mean cross-entropy objective per preference."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [num_preferences, batch, classes]")
    count, batch_size, num_classes = logits.shape
    if labels.ndim != 1 or labels.numel() != batch_size:
        raise ValueError("labels must have shape [batch]")
    expanded_labels = labels.unsqueeze(0).expand(count, batch_size).reshape(-1)
    losses = F.cross_entropy(
        logits.reshape(-1, num_classes), expanded_labels, reduction="none"
    )
    return losses.reshape(count, batch_size).mean(dim=1)


def demographic_parity_surrogate(
    logits: torch.Tensor, protected: torch.Tensor
) -> torch.Tensor:
    """PraFFL's differentiable squared centered-product DP surrogate."""
    if logits.ndim != 3 or logits.shape[1] != protected.numel():
        raise ValueError("logits and protected batch dimensions must match")
    sensitive = protected.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    centered_sensitive = sensitive - sensitive.mean()
    centered_logits = logits - logits.mean(dim=1, keepdim=True)
    products = centered_sensitive[None, :, None] * centered_logits
    return products.square().mean(dim=1).sum(dim=1)


def smooth_tchebycheff(
    accuracy_loss: torch.Tensor,
    fairness_loss: torch.Tensor,
    preferences: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Inverse-preference smooth Tchebycheff scalarization from PraFFL."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if preferences.ndim != 2 or preferences.shape[1] != 2:
        raise ValueError("preferences must have shape [num_preferences, 2]")
    if torch.any(preferences <= 0):
        raise ValueError("training preferences must be strictly positive")
    if accuracy_loss.shape != fairness_loss.shape or accuracy_loss.ndim != 1:
        raise ValueError("objective losses must be matching one-dimensional tensors")
    if accuracy_loss.numel() != preferences.shape[0]:
        raise ValueError("objective and preference counts must match")
    objectives = torch.stack(
        (
            accuracy_loss / preferences[:, 0],
            fairness_loss / preferences[:, 1],
        ),
        dim=1,
    )
    return torch.logsumexp(gamma * objectives, dim=1) / gamma


def clone_state_dict_to_cpu(module: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Take an independent, detached CPU snapshot suitable for private state."""
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in module.state_dict().items()
    )


def uniform_average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> OrderedDict[str, torch.Tensor]:
    """Uniformly average communicated encoder states (PraFFL equation 14)."""
    if not states:
        raise ValueError("cannot average an empty state list")
    reference_keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != reference_keys for state in states[1:]):
        raise ValueError("all state dictionaries must have identical ordered keys")
    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in reference_keys:
        tensors = [state[name].detach().cpu() for state in states]
        reference = tensors[0]
        if any(
            tensor.shape != reference.shape or tensor.dtype != reference.dtype
            for tensor in tensors[1:]
        ):
            raise ValueError(f"state tensor {name!r} has incompatible shape or dtype")
        if reference.is_floating_point() or reference.is_complex():
            averaged[name] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError(f"non-floating tensor {name!r} differs across clients")
            averaged[name] = reference.clone()
    return averaged
