"""Local optimization phases for the paper-faithful PraFFL adaptation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from algorithm.praffl_core import (
    HyperNetwork,
    PraFFLConfig,
    clone_state_dict_to_cpu,
    demographic_parity_surrogate,
    functional_linear_heads,
    preference_cross_entropy,
    smooth_tchebycheff,
)
from tool.amp_utils import autocast_context, scale_backward, scaler_step


PreferenceSampler = Callable[[int, torch.device, torch.dtype], torch.Tensor]


@dataclass(frozen=True)
class ClientTrainResult:
    encoder_state: dict[str, torch.Tensor]
    hypernetwork_state: dict[str, torch.Tensor]
    communicated_losses: tuple[float, ...]
    personalized_losses: tuple[float, ...]
    gpu_seconds: float


def make_optimizer(parameters, optimizer_name: str, learning_rate: float):
    parameters = list(parameters)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"unsupported PraFFL optimizer {optimizer_name!r}")


def sample_dirichlet_preferences(
    count: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    concentration = torch.ones(2, device=device, dtype=dtype)
    return torch.distributions.Dirichlet(concentration).sample((count,))


def _move_batch(batch, device: torch.device):
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device).long(),
        batch["protected"].to(device),
    )


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)
        parameter.grad = None


def _optimizer_step(loss, optimizer, scaler) -> None:
    optimizer.zero_grad(set_to_none=True)
    scale_backward(loss, scaler)
    scaler_step(scaler, optimizer)


def train_communicated_phase(
    model: torch.nn.Module,
    hypernetwork: HyperNetwork,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    epochs: int,
    optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
) -> list[float]:
    model.bert.train()
    model.drop.train()
    model.out.eval()
    hypernetwork.eval()
    _set_requires_grad(model.bert, True)
    _set_requires_grad(model.out, False)
    _set_requires_grad(hypernetwork, False)
    balanced = torch.tensor([[0.5, 0.5]], device=device)
    with torch.no_grad():
        fixed_weight, fixed_bias = hypernetwork(balanced)
        fixed_weight = fixed_weight.detach()
        fixed_bias = fixed_bias.detach()
    losses: list[float] = []
    for _epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels, _protected = _move_batch(batch, device)
            with autocast_context(device, use_amp):
                features = model.drop(model.encode(input_ids, attention_mask))
                logits = functional_linear_heads(features, fixed_weight, fixed_bias)
                loss = preference_cross_entropy(logits, labels).mean()
            _optimizer_step(loss, optimizer, scaler)
            losses.append(float(loss.detach().cpu()))
    return losses


def train_personalized_phase(
    model: torch.nn.Module,
    hypernetwork: HyperNetwork,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    epochs: int,
    preference_batch_size: int,
    smooth_gamma: float,
    optimizer,
    device: torch.device,
    use_amp: bool,
    scaler,
    preference_sampler: PreferenceSampler = sample_dirichlet_preferences,
) -> list[float]:
    model.bert.eval()
    model.drop.eval()
    model.out.eval()
    hypernetwork.train()
    _set_requires_grad(model.bert, False)
    _set_requires_grad(model.out, False)
    _set_requires_grad(hypernetwork, True)
    losses: list[float] = []
    for _epoch in range(epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels, protected = _move_batch(batch, device)
            with torch.no_grad():
                features = model.drop(model.encode(input_ids, attention_mask)).detach()
            preferences = preference_sampler(
                preference_batch_size,
                device,
                torch.float32,
            )
            with autocast_context(device, use_amp):
                weight, bias = hypernetwork(preferences)
                logits = functional_linear_heads(features, weight, bias)
                accuracy_loss = preference_cross_entropy(logits.float(), labels)
                fairness_loss = demographic_parity_surrogate(
                    logits.float(), protected
                )
                loss = smooth_tchebycheff(
                    accuracy_loss,
                    fairness_loss,
                    preferences.float(),
                    gamma=smooth_gamma,
                ).mean()
            _optimizer_step(loss, optimizer, scaler)
            losses.append(float(loss.detach().cpu()))
    return losses


def train_praffl_client(
    global_model: torch.nn.Module,
    hypernetwork_template: HyperNetwork,
    private_hypernetwork_state: dict[str, torch.Tensor],
    dataloader,
    config: PraFFLConfig,
    device: torch.device,
    use_amp: bool,
    scaler,
) -> ClientTrainResult:
    local_model = copy.deepcopy(global_model).to(device)
    local_hypernetwork = copy.deepcopy(hypernetwork_template).to(device)
    local_hypernetwork.load_state_dict(private_hypernetwork_state, strict=True)
    encoder_optimizer = make_optimizer(
        local_model.bert.parameters(),
        config.optimizer_name,
        config.encoder_learning_rate,
    )
    hypernetwork_optimizer = make_optimizer(
        local_hypernetwork.parameters(),
        "adam",
        config.hypernetwork_learning_rate,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    communicated_losses = train_communicated_phase(
        local_model,
        local_hypernetwork,
        dataloader,
        epochs=config.tau_c,
        optimizer=encoder_optimizer,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    personalized_losses = train_personalized_phase(
        local_model,
        local_hypernetwork,
        dataloader,
        epochs=config.tau_p,
        preference_batch_size=config.preference_batch_size,
        smooth_gamma=config.smooth_gamma,
        optimizer=hypernetwork_optimizer,
        device=device,
        use_amp=use_amp,
        scaler=scaler,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = ClientTrainResult(
        encoder_state=clone_state_dict_to_cpu(local_model.bert),
        hypernetwork_state=clone_state_dict_to_cpu(local_hypernetwork),
        communicated_losses=tuple(communicated_losses),
        personalized_losses=tuple(personalized_losses),
        gpu_seconds=elapsed,
    )
    del encoder_optimizer, hypernetwork_optimizer, local_hypernetwork, local_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
