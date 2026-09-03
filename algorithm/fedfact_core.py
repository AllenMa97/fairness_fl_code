from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch

from module.partition import extract_dataset_view


FEDFACT_STATE_SCHEMA_VERSION = 1


class SupportError(ValueError):
    pass


@dataclass(frozen=True)
class FedFACTConfig:
    task: str
    num_clients: int
    fraction: float
    drop_rate: float
    fairness_metric: Literal["DP", "EO"]
    global_constraint: float
    local_constraint: float
    dual_learning_rate: float
    dual_bound: float
    dual_init: float
    ensemble_learning_rate: float
    ensemble_weight_init: float
    calibration_epsilon: float

    @property
    def num_constraints(self) -> int:
        return 1 if self.fairness_metric == "DP" else 2

    @classmethod
    def from_param_dict(cls, params: Mapping) -> "FedFACTConfig":
        metric = str(params["fairness_metric"]).upper()
        config = cls(
            task=str(params["task"]),
            num_clients=int(params["num_clients_K"]),
            fraction=float(params["FL_fraction"]),
            drop_rate=float(params["FL_drop_rate"]),
            fairness_metric=metric,
            global_constraint=float(params["global_constraint"]),
            local_constraint=float(params["local_constraint"]),
            dual_learning_rate=float(params["dual_learning_rate"]),
            dual_bound=float(params["dual_bound"]),
            dual_init=float(params["dual_init"]),
            ensemble_learning_rate=float(params["ensemble_learning_rate"]),
            ensemble_weight_init=float(params["ensemble_weight_init"]),
            calibration_epsilon=float(params["calibration_epsilon"]),
        )
        numeric = (
            config.fraction, config.drop_rate, config.global_constraint,
            config.local_constraint, config.dual_learning_rate,
            config.dual_bound, config.dual_init,
            config.ensemble_learning_rate, config.ensemble_weight_init,
            config.calibration_epsilon,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("FedFACT configuration values must be finite")
        if config.task != "SENT_CLF":
            raise ValueError("paper-faithful FedFACT only supports SENT_CLF")
        if config.num_clients <= 0:
            raise ValueError("num_clients_K must be positive")
        if config.fraction != 1.0:
            raise ValueError("paper-faithful FedFACT requires FL_fraction == 1.0")
        if config.drop_rate != 0.0:
            raise ValueError("paper-faithful FedFACT requires FL_drop_rate == 0.0")
        if metric not in {"DP", "EO"}:
            raise ValueError("FedFACT fairness_metric must be DP or EO")
        if config.global_constraint < 0 or config.local_constraint < 0:
            raise ValueError("FedFACT global/local constraints must be nonnegative")
        if config.dual_learning_rate <= 0 or config.dual_bound <= 0:
            raise ValueError("FedFACT dual learning rate/bound must be positive")
        if config.dual_init < 0:
            raise ValueError("FedFACT dual_init must be nonnegative")
        if 2 * config.num_constraints * config.dual_init > config.dual_bound:
            raise ValueError("FedFACT initial dual is outside the L1 bound")
        if config.ensemble_learning_rate <= 0:
            raise ValueError("FedFACT ensemble_learning_rate must be positive")
        if not 0 < config.ensemble_weight_init < 1:
            raise ValueError("FedFACT ensemble_weight_init must be in (0,1)")
        if config.calibration_epsilon <= 0:
            raise ValueError("FedFACT calibration_epsilon must be positive")
        return config


@dataclass(frozen=True)
class SupportStatistics:
    counts: torch.Tensor
    client_totals: torch.Tensor


def _validate_metric(metric: str) -> str:
    metric = str(metric).upper()
    if metric not in {"DP", "EO"}:
        raise ValueError("FedFACT fairness metric must be DP or EO")
    return metric


def build_support_statistics(datasets: Sequence, metric: str) -> SupportStatistics:
    metric = _validate_metric(metric)
    if not datasets:
        raise SupportError("FedFACT requires at least one client dataset")
    counts = torch.zeros((len(datasets), 2, 2), dtype=torch.float64)
    for client_id, dataset in enumerate(datasets):
        view = extract_dataset_view(dataset)
        for label, protected in zip(view.labels.tolist(), view.protected.tolist()):
            if label not in (0, 1) or protected not in (0, 1):
                raise SupportError(
                    f"FedFACT requires binary values; client {client_id} has "
                    f"label={label!r}, protected={protected!r}"
                )
            counts[client_id, int(protected), int(label)] += 1
        for protected in range(2):
            if counts[client_id, protected].sum() == 0:
                raise SupportError(
                    f"FedFACT DP support missing: client {client_id}, protected={protected}"
                )
            if metric == "EO":
                for label in range(2):
                    if counts[client_id, protected, label] == 0:
                        raise SupportError(
                            f"FedFACT EO support missing: client {client_id}, "
                            f"protected={protected}, label={label}"
                        )
    return SupportStatistics(counts=counts, client_totals=counts.sum(dim=(1, 2)))


def project_nonnegative_l1_ball(values: torch.Tensor, bound: float) -> torch.Tensor:
    if not math.isfinite(float(bound)) or bound <= 0:
        raise ValueError("L1 bound must be finite and positive")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("projection values must be finite")
    shape = tensor.shape
    flat = tensor.flatten().clamp_min(0)
    if flat.sum().item() <= bound:
        return flat.reshape(shape)
    ordered = torch.sort(flat, descending=True).values
    cssv = torch.cumsum(ordered, 0) - bound
    positions = torch.arange(1, flat.numel() + 1, dtype=torch.float64, device=flat.device)
    active = ordered - cssv / positions > 0
    rho = torch.nonzero(active, as_tuple=False)[-1, 0]
    threshold = cssv[rho] / positions[rho]
    return (flat - threshold).clamp_min(0).reshape(shape)


def _cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def initialize_fedfact_state(
    model: torch.nn.Module,
    num_clients: int,
    metric: str,
    stats: SupportStatistics,
    dual_init: float,
    ensemble_weight_init: float,
) -> dict:
    metric = _validate_metric(metric)
    constraints = 1 if metric == "DP" else 2
    base = _cpu_model_state(model)
    return {
        "schema_version": FEDFACT_STATE_SCHEMA_VERSION,
        "variant": "fedfact_in",
        "fairness_metric": metric,
        "personal_model_states": [copy.deepcopy(base) for _ in range(num_clients)],
        "global_dual": torch.full((constraints, 2), dual_init, dtype=torch.float64),
        "local_duals": torch.full((num_clients, constraints, 2), dual_init, dtype=torch.float64),
        "ensemble_weights": torch.full((num_clients,), ensemble_weight_init, dtype=torch.float64),
        "support_counts": stats.counts.detach().cpu().clone().to(torch.float64),
        "client_sample_counts": stats.client_totals.detach().cpu().clone().to(torch.float64),
    }


def _clone_and_validate_tensor(value, name: str, shape: tuple[int, ...], dtype=torch.float64):
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"FedFACT state {name} has incompatible shape")
    if value.device.type != "cpu" or value.dtype != dtype:
        raise ValueError(f"FedFACT state {name} must be CPU {dtype}")
    if not torch.isfinite(value).all():
        raise ValueError(f"FedFACT state {name} must be finite")
    return value.clone()


def validate_fedfact_state(
    state: Mapping,
    model: torch.nn.Module,
    num_clients: int,
    metric: str,
    stats: SupportStatistics,
    dual_bound: float,
) -> dict:
    metric = _validate_metric(metric)
    if state.get("schema_version") != FEDFACT_STATE_SCHEMA_VERSION:
        raise ValueError("incompatible FedFACT algorithm-state schema")
    if state.get("variant") != "fedfact_in" or state.get("fairness_metric") != metric:
        raise ValueError("FedFACT state variant/fairness metric mismatch")
    constraints = 1 if metric == "DP" else 2
    global_dual = _clone_and_validate_tensor(state.get("global_dual"), "global_dual", (constraints, 2))
    local_duals = _clone_and_validate_tensor(state.get("local_duals"), "local_duals", (num_clients, constraints, 2))
    weights = _clone_and_validate_tensor(state.get("ensemble_weights"), "ensemble_weights", (num_clients,))
    support = _clone_and_validate_tensor(state.get("support_counts"), "support_counts", (num_clients, 2, 2))
    totals = _clone_and_validate_tensor(state.get("client_sample_counts"), "client_sample_counts", (num_clients,))
    if (global_dual < 0).any() or (local_duals < 0).any():
        raise ValueError("FedFACT dual state must be nonnegative")
    if global_dual.sum().item() > dual_bound + 1e-12:
        raise ValueError("FedFACT global dual exceeds L1 bound")
    if any(local_duals[k].sum().item() > dual_bound + 1e-12 for k in range(num_clients)):
        raise ValueError("FedFACT local dual exceeds L1 bound")
    if ((weights <= 0) | (weights >= 1)).any():
        raise ValueError("FedFACT ensemble weights must be in (0,1)")
    if not torch.equal(support, stats.counts) or not torch.equal(totals, stats.client_totals):
        raise ValueError("FedFACT support counts do not match current partition")
    personal = state.get("personal_model_states")
    if not isinstance(personal, (list, tuple)) or len(personal) != num_clients:
        raise ValueError("FedFACT state must contain one personal model per client")
    reference = model.state_dict()
    cloned_personal = []
    for client_id, candidate in enumerate(personal):
        if not isinstance(candidate, Mapping) or set(candidate) != set(reference):
            raise ValueError(f"FedFACT personal model {client_id} keys mismatch")
        clone = {}
        for name, expected in reference.items():
            value = candidate[name]
            if not torch.is_tensor(value) or value.device.type != "cpu":
                raise ValueError(f"FedFACT personal model {client_id}.{name} must be a CPU tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"FedFACT personal model {client_id}.{name} shape/dtype mismatch")
            clone[name] = value.clone()
        cloned_personal.append(clone)
    return {
        "schema_version": FEDFACT_STATE_SCHEMA_VERSION,
        "variant": "fedfact_in",
        "fairness_metric": metric,
        "personal_model_states": cloned_personal,
        "global_dual": global_dual,
        "local_duals": local_duals,
        "ensemble_weights": weights,
        "support_counts": support,
        "client_sample_counts": totals,
    }


def build_calibration_matrices(
    client_id: int,
    support_counts: torch.Tensor,
    metric: str,
    global_dual: torch.Tensor,
    local_dual: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    metric = _validate_metric(metric)
    counts = torch.as_tensor(support_counts, dtype=torch.float64, device="cpu")
    if counts.ndim != 3 or tuple(counts.shape[1:]) != (2, 2):
        raise ValueError("support_counts must have shape [client,2,2]")
    if not 0 <= client_id < counts.shape[0]:
        raise ValueError("invalid FedFACT client id")
    constraints = 1 if metric == "DP" else 2
    global_dual = torch.as_tensor(global_dual, dtype=torch.float64, device="cpu")
    local_dual = torch.as_tensor(local_dual, dtype=torch.float64, device="cpu")
    if global_dual.shape != (constraints, 2) or local_dual.shape != (constraints, 2):
        raise ValueError("FedFACT calibration dual shape mismatch")
    if not torch.isfinite(counts).all() or not torch.isfinite(global_dual).all() or not torch.isfinite(local_dual).all():
        raise ValueError("FedFACT calibration inputs must be finite")
    if epsilon <= 0 or not math.isfinite(float(epsilon)):
        raise ValueError("FedFACT calibration epsilon must be positive")
    total = counts.sum()
    if total <= 0:
        raise SupportError("FedFACT calibration support is empty")
    matrices = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
    for a in (0, 1):
        sign = 2 * a - 1
        n_a_k = counts[client_id, a].sum()
        p_a_k = n_a_k / total
        if p_a_k <= 0:
            raise SupportError(f"FedFACT calibration missing client {client_id}, protected={a}")
        if metric == "DP":
            global_group = counts[:, a, :].sum()
            if global_group <= 0:
                raise SupportError(f"FedFACT calibration missing protected={a}")
            p_k_given_a = n_a_k / global_group
            d_global = sign * p_k_given_a
            d_local = float(sign)
            correction = (
                (global_dual[0, 0] - global_dual[0, 1]) * d_global
                + (local_dual[0, 0] - local_dual[0, 1]) * d_local
            ) / p_a_k
            matrices[a, :, 1] -= correction
        else:
            for y in (0, 1):
                p_a_y = counts[:, a, y].sum() / total
                p_a_y_k = counts[client_id, a, y] / total
                if p_a_y <= 0 or p_a_y_k <= 0:
                    raise SupportError(
                        f"FedFACT calibration missing client {client_id}, protected={a}, label={y}"
                    )
                d_global = sign * p_a_k / p_a_y
                d_local = sign * p_a_k / p_a_y_k
                correction = (
                    (global_dual[y, 0] - global_dual[y, 1]) * d_global
                    + (local_dual[y, 0] - local_dual[y, 1]) * d_local
                ) / p_a_k
                matrices[a, y, 1] -= correction
    kappa = max(0.0, float(epsilon) - matrices.min().item())
    return matrices + kappa


def calibrated_loss(logits, labels, protected, matrices):
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("FedFACT requires two-logit model output")
    labels = labels.reshape(-1).long()
    protected = protected.reshape(-1).long()
    if labels.numel() != logits.shape[0] or protected.numel() != logits.shape[0]:
        raise ValueError("FedFACT batch dimensions do not match")
    if ((labels < 0) | (labels > 1)).any() or ((protected < 0) | (protected > 1)).any():
        raise ValueError("FedFACT requires binary labels and protected values")
    matrix = torch.as_tensor(matrices, device=logits.device, dtype=torch.float32)
    if matrix.shape != (2, 2, 2) or not torch.isfinite(matrix).all():
        raise ValueError("FedFACT calibration matrices must be finite [2,2,2]")
    selected = matrix[protected, labels, :]
    log_probabilities = torch.log_softmax(logits.float(), dim=1)
    return -(selected * log_probabilities).sum(dim=1).mean()


def ensemble_probabilities(theta_logits, phi_logits, weight):
    if theta_logits.shape != phi_logits.shape or theta_logits.ndim != 2 or theta_logits.shape[-1] != 2:
        raise ValueError("FedFACT ensemble requires matching two-logit tensors")
    w = torch.as_tensor(weight, device=theta_logits.device, dtype=torch.float32)
    if w.numel() != 1 or not torch.isfinite(w) or not bool((w > 0) & (w < 1)):
        raise ValueError("FedFACT ensemble weight must be in (0,1)")
    return w * torch.softmax(theta_logits.float(), 1) + (1 - w) * torch.softmax(phi_logits.float(), 1)


def update_ensemble_weight(weight, theta_loss, phi_loss, learning_rate):
    w = torch.as_tensor(weight, dtype=torch.float64)
    theta_loss = torch.as_tensor(theta_loss, dtype=torch.float64)
    phi_loss = torch.as_tensor(phi_loss, dtype=torch.float64)
    eta = torch.as_tensor(learning_rate, dtype=torch.float64)
    if w.numel() != 1 or not torch.isfinite(torch.stack((w, theta_loss, phi_loss, eta))).all():
        raise ValueError("FedFACT weight update inputs must be finite scalars")
    if not 0 < w.item() < 1 or eta.item() <= 0:
        raise ValueError("invalid FedFACT weight update")
    log_odds = torch.log(w) - torch.log1p(-w)
    new_weight = torch.sigmoid(log_odds + eta * (phi_loss - theta_loss))
    eps = torch.finfo(torch.float64).eps
    return new_weight.clamp(eps, 1 - eps)


def confusion_from_predictions(predictions, labels, protected):
    predictions = predictions.detach().cpu().reshape(-1).long()
    labels = labels.detach().cpu().reshape(-1).long()
    protected = protected.detach().cpu().reshape(-1).long()
    if not (predictions.numel() == labels.numel() == protected.numel()):
        raise ValueError("FedFACT prediction dimensions do not match")
    for name, values in (("predictions", predictions), ("labels", labels), ("protected", protected)):
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"FedFACT {name} must be binary")
    flat = protected * 4 + labels * 2 + predictions
    return torch.bincount(flat, minlength=8).reshape(2, 2, 2).to(torch.float64)


def disparity_from_confusion(confusion, metric):
    metric = _validate_metric(metric)
    confusion = torch.as_tensor(confusion, dtype=torch.float64, device="cpu")
    if confusion.shape != (2, 2, 2) or not torch.isfinite(confusion).all() or (confusion < 0).any():
        raise ValueError("FedFACT confusion must be finite nonnegative [2,2,2]")
    if metric == "DP":
        denominator = confusion.sum(dim=(1, 2))
        if (denominator == 0).any():
            raise SupportError("DP disparity requires both protected groups")
        rate = confusion[:, :, 1].sum(1) / denominator
        return (rate[1] - rate[0]).reshape(1)
    denominator = confusion.sum(dim=2)
    if (denominator == 0).any():
        raise SupportError("EO disparity requires every protected/label cell")
    rate = confusion[:, :, 1] / denominator
    return rate[1, :] - rate[0, :]


def update_dual(dual, disparity, tolerance, learning_rate, bound):
    dual = torch.as_tensor(dual, dtype=torch.float64, device="cpu")
    disparity = torch.as_tensor(disparity, dtype=torch.float64, device="cpu").reshape(-1)
    values = (disparity, torch.tensor(float(tolerance)), torch.tensor(float(learning_rate)), torch.tensor(float(bound)))
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError("FedFACT dual update inputs must be finite")
    if tolerance < 0 or learning_rate <= 0 or bound <= 0:
        raise ValueError("invalid FedFACT dual update hyperparameters")
    residual = torch.stack((disparity - tolerance, -disparity - tolerance), dim=1)
    if residual.shape != dual.shape:
        raise ValueError("dual/disparity shape mismatch")
    return project_nonnegative_l1_ball(dual + learning_rate * residual, bound)
