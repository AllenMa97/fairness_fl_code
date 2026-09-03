from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from algorithm.praffl_core import (
    PRAFFL_STATE_SCHEMA_VERSION,
    HyperNetwork,
    functional_linear_heads,
)
from tool.amp_utils import autocast_context


class PraFFLEvaluationError(ValueError):
    """Raised when PraFFL metrics would be scientifically undefined."""


def build_preference_grid(num_points: int) -> torch.Tensor:
    if num_points < 2:
        raise ValueError("PraFFL evaluation needs at least two preference points")
    accuracy_weight = torch.linspace(0.0, 1.0, steps=num_points, dtype=torch.float32)
    return torch.stack((accuracy_weight, 1.0 - accuracy_weight), dim=1)


def metrics_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    protected: torch.Tensor,
    *,
    scope: str,
) -> dict[str, torch.Tensor]:
    predictions = predictions.detach().cpu().long()
    labels = labels.detach().cpu().long().reshape(-1)
    protected = protected.detach().cpu().long().reshape(-1)
    if predictions.ndim != 2 or predictions.shape[1] != labels.numel():
        raise ValueError("predictions must have shape [preferences, examples]")
    if labels.numel() == 0:
        raise PraFFLEvaluationError(f"{scope} has no examples")
    if not torch.all((labels == 0) | (labels == 1)):
        raise PraFFLEvaluationError(f"{scope} has non-binary labels")
    if not torch.all((protected == 0) | (protected == 1)):
        raise PraFFLEvaluationError(f"{scope} has protected values outside 0/1")
    group_counts = torch.stack(((protected == 0).sum(), (protected == 1).sum())).to(torch.float64)
    if torch.any(group_counts == 0):
        raise PraFFLEvaluationError(
            f"{scope} protected counts are {group_counts.to(torch.int64).tolist()}; both groups are required"
        )

    accuracy = (predictions == labels.unsqueeze(0)).to(torch.float64).mean(dim=1)
    positive_rates = []
    true_positive_rates = []
    for group in (0, 1):
        group_mask = protected == group
        positive_rates.append(
            predictions[:, group_mask].to(torch.float64).mean(dim=1)
        )
        positive_label_mask = group_mask & (labels == 1)
        if positive_label_mask.any():
            true_positive_rates.append(
                predictions[:, positive_label_mask].to(torch.float64).mean(dim=1)
            )
        else:
            true_positive_rates.append(torch.zeros(predictions.shape[0], dtype=torch.float64))
    rate_0, rate_1 = positive_rates
    tpr_0, tpr_1 = true_positive_rates
    overall_positive = predictions.to(torch.float64).mean(dim=1)
    dp_disparity = torch.maximum(
        (rate_0 - overall_positive).abs(),
        (rate_1 - overall_positive).abs(),
    )
    return {
        "ACC": accuracy,
        "DEO": (tpr_0 - tpr_1).abs(),
        "SPD": rate_0 - rate_1,
        "DP": dp_disparity,
    }


def evaluate_preference_grid(
    model: torch.nn.Module,
    hypernetwork: HyperNetwork,
    dataloader: Iterable[dict[str, torch.Tensor]],
    preferences: torch.Tensor,
    *,
    device: torch.device,
    use_amp: bool,
    chunk_size: int,
    scope: str,
) -> dict[str, torch.Tensor]:
    if preferences.ndim != 2 or preferences.shape[1] != 2:
        raise ValueError("evaluation preferences must have shape [points, 2]")
    if chunk_size < 1:
        raise ValueError("preference chunk size must be positive")
    model.bert.eval()
    model.drop.eval()
    model.out.eval()
    hypernetwork.eval()
    prediction_batches = [[] for _ in range(preferences.shape[0])]
    label_batches = []
    protected_batches = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].detach().cpu().long()
            protected = batch["protected"].detach().cpu().long()
            with autocast_context(device, use_amp):
                features = model.drop(model.encode(input_ids, attention_mask))
            for start in range(0, preferences.shape[0], chunk_size):
                stop = min(start + chunk_size, preferences.shape[0])
                chunk = preferences[start:stop].to(device=device, dtype=features.dtype)
                with autocast_context(device, use_amp):
                    weight, bias = hypernetwork(chunk)
                    logits = functional_linear_heads(features, weight, bias)
                predicted = logits.argmax(dim=2).detach().cpu()
                for offset, row in enumerate(predicted):
                    prediction_batches[start + offset].append(row)
            label_batches.append(labels)
            protected_batches.append(protected)
    if not label_batches:
        raise PraFFLEvaluationError(f"{scope} has no batches")
    predictions = torch.stack(
        [torch.cat(rows, dim=0) for rows in prediction_batches],
        dim=0,
    )
    return metrics_from_predictions(
        predictions,
        torch.cat(label_batches, dim=0),
        torch.cat(protected_batches, dim=0),
        scope=scope,
    )


def pareto_front_2d(points: Sequence[Sequence[float]]) -> list[list[float]]:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    front = []
    for candidate in unique:
        dominated = any(
            other[0] <= candidate[0]
            and other[1] <= candidate[1]
            and other != candidate
            for other in unique
        )
        if not dominated:
            front.append([candidate[0], candidate[1]])
    return front


def hypervolume_2d(
    pareto_points: Sequence[Sequence[float]],
    *,
    reference_point: tuple[float, float],
) -> float:
    reference_x, reference_y = reference_point
    front = pareto_front_2d(pareto_points)
    current_y = reference_y
    hypervolume = 0.0
    for x_value, y_value in front:
        if x_value < reference_x and y_value < current_y:
            hypervolume += (reference_x - x_value) * (current_y - y_value)
            current_y = y_value
    return float(hypervolume)


def _client_output(
    metrics: Mapping[str, torch.Tensor],
    preferences: torch.Tensor,
    grid_size: int,
) -> dict:
    solutions = []
    objective_points = []
    for index in range(grid_size):
        accuracy = float(metrics["ACC"][index])
        dp_disparity = float(metrics["DP"][index])
        objectives = [1.0 - accuracy, dp_disparity]
        objective_points.append(objectives)
        solutions.append(
            {
                "preference": [float(value) for value in preferences[index]],
                "ACC": accuracy,
                "DP": dp_disparity,
                "objectives": objectives,
            }
        )
    front = pareto_front_2d(objective_points)
    return {
        "solutions": solutions,
        "pareto_front": front,
        "hv": hypervolume_2d(front, reference_point=(1.0, 1.0)),
    }


def _validate_evaluator_state(algorithm_state: Mapping[str, object], num_clients: int) -> tuple[dict, Mapping]:
    if algorithm_state.get("schema_version") != PRAFFL_STATE_SCHEMA_VERSION:
        raise PraFFLEvaluationError("incompatible PraFFL algorithm-state schema")
    spec = algorithm_state.get("hypernetwork_spec")
    private_states = algorithm_state.get("client_hypernetworks")
    if not isinstance(spec, dict):
        raise PraFFLEvaluationError("PraFFL state is missing hypernetwork_spec")
    if not isinstance(private_states, Mapping) or set(private_states) != set(range(num_clients)):
        raise PraFFLEvaluationError("PraFFL state must contain every private client hypernetwork")
    return spec, private_states


def _load_private_hypernetwork(spec, private_state, device):
    hypernetwork = HyperNetwork(
        preference_dim=int(spec["preference_dim"]),
        feature_dim=int(spec["feature_dim"]),
        num_classes=int(spec["num_classes"]),
        hidden_dim=int(spec["hidden_dim"]),
    )
    hypernetwork.load_state_dict(private_state, strict=True)
    return hypernetwork.to(device)


def evaluate_praffl_report(
    global_model,
    param_dict,
    testing_dataloader,
    algorithm_state,
) -> dict[str, float]:
    """Evaluate the named comparison preference after a communication round."""
    num_clients = int(param_dict["num_clients_K"])
    spec, private_states = _validate_evaluator_state(algorithm_state, num_clients)
    report = torch.tensor(
        param_dict.get("praffl_report_preference", (0.5, 0.5)),
        dtype=torch.float32,
    ).reshape(1, 2)
    if torch.any(report < 0) or not torch.isclose(
        report.sum(), torch.tensor(1.0)
    ):
        raise PraFFLEvaluationError(
            "praffl_report_preference must be non-negative and sum to 1"
        )
    device = torch.device(param_dict["device"])
    use_amp = bool(param_dict.get("use_amp", False))
    global_model = global_model.to(device)
    values = {"ACC": [], "DEO": [], "SPD": []}
    for client_id in range(num_clients):
        hypernetwork = _load_private_hypernetwork(
            spec, private_states[client_id], device
        )
        metrics = evaluate_preference_grid(
            global_model,
            hypernetwork,
            testing_dataloader,
            report,
            device=device,
            use_amp=use_amp,
            chunk_size=1,
            scope=f"client {client_id} round global test",
        )
        for name in values:
            values[name].append(float(metrics[name][0]))
        del hypernetwork
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {name: float(np.mean(items)) for name, items in values.items()}


def evaluate_praffl(global_model, param_dict, data_bundle, algorithm_state) -> dict:
    num_clients = int(param_dict["num_clients_K"])
    if len(data_bundle.client_testing_dataloaders) != num_clients:
        raise PraFFLEvaluationError("PraFFL needs one client test loader per client")
    spec, private_states = _validate_evaluator_state(algorithm_state, num_clients)
    grid_size = int(param_dict.get("praffl_preference_points", 1000))
    chunk_size = int(param_dict.get("praffl_preference_chunk_size", 128))
    report_raw = param_dict.get("praffl_report_preference", (0.5, 0.5))
    report = torch.tensor(report_raw, dtype=torch.float32).reshape(1, 2)
    if torch.any(report < 0) or not torch.isclose(report.sum(), torch.tensor(1.0)):
        raise PraFFLEvaluationError("praffl_report_preference must be non-negative and sum to 1")
    grid = build_preference_grid(grid_size)
    evaluation_preferences = torch.cat((grid, report), dim=0)
    device = torch.device(param_dict["device"])
    use_amp = bool(param_dict.get("use_amp", False))
    global_model = global_model.to(device)

    local_clients = {}
    global_clients = {}
    local_hv = []
    global_hv = []
    report_metrics = {"ACC": [], "DEO": [], "SPD": []}
    for client_id in range(num_clients):
        hypernetwork = _load_private_hypernetwork(
            spec, private_states[client_id], device
        )
        local_metrics = evaluate_preference_grid(
            global_model,
            hypernetwork,
            data_bundle.client_testing_dataloaders[client_id],
            evaluation_preferences,
            device=device,
            use_amp=use_amp,
            chunk_size=chunk_size,
            scope=f"client {client_id} local test",
        )
        global_metrics = evaluate_preference_grid(
            global_model,
            hypernetwork,
            data_bundle.testing_dataloader,
            evaluation_preferences,
            device=device,
            use_amp=use_amp,
            chunk_size=chunk_size,
            scope=f"client {client_id} global test",
        )
        local_output = _client_output(local_metrics, grid, grid_size)
        global_output = _client_output(global_metrics, grid, grid_size)
        local_clients[str(client_id)] = local_output
        global_clients[str(client_id)] = global_output
        local_hv.append(local_output["hv"])
        global_hv.append(global_output["hv"])
        for metric_name in report_metrics:
            report_metrics[metric_name].append(float(global_metrics[metric_name][grid_size]))
        del hypernetwork
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        metric_name: float(np.mean(values))
        for metric_name, values in report_metrics.items()
    }
    output["report_preference"] = [float(report[0, 0]), float(report[0, 1])]
    output["praffl"] = {
        "preference_grid": [[float(value) for value in row] for row in grid],
        "reference_point": [1.0, 1.0],
        "report_scope": "mean of private client heads on the common global test loader",
        "local": {
            "mean_hv": float(np.mean(local_hv)),
            "clients": local_clients,
        },
        "global": {
            "mean_hv": float(np.mean(global_hv)),
            "clients": global_clients,
        },
    }
    global_model.cpu()
    return output
