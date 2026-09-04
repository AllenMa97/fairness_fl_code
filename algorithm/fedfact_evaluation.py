from __future__ import annotations

import copy
import statistics

import torch

from algorithm.FedFACT import _forward_text_logits
from algorithm.fedfact_core import (
    FedFACTConfig,
    SupportError,
    build_support_statistics,
    confusion_from_predictions,
    disparity_from_confusion,
    ensemble_probabilities,
    validate_fedfact_state,
)


def _harmonic_mean(left: float, right: float) -> float:
    return 0.0 if left + right == 0 else 2 * left * right / (left + right)


def _client_confusion(theta, phi, loader, weight, device):
    theta.eval()
    phi.eval()
    predictions, labels, protected = [], [], []
    with torch.no_grad():
        for batch in loader:
            theta_logits = _forward_text_logits(theta, batch, device)
            phi_logits = _forward_text_logits(phi, batch, device)
            probabilities = ensemble_probabilities(theta_logits, phi_logits, weight)
            predictions.append(probabilities.argmax(1).cpu())
            labels.append(batch["labels"].reshape(-1).cpu())
            protected.append(batch["protected"].reshape(-1).cpu())
    if not predictions:
        raise SupportError("FedFACT test loader is empty")
    return confusion_from_predictions(
        torch.cat(predictions), torch.cat(labels), torch.cat(protected)
    )


def _metrics_from_confusions(confusions, config):
    global_confusion = torch.stack(confusions).sum(0)
    total = global_confusion.sum().item()
    if total <= 0:
        raise SupportError("FedFACT evaluation has no samples")
    accuracy = float(
        (global_confusion[:, 0, 0].sum() + global_confusion[:, 1, 1].sum()).item()
        / total
    )
    global_dp = disparity_from_confusion(global_confusion, "DP")
    selected_global = disparity_from_confusion(global_confusion, config.fairness_metric)
    local_disparities = [
        disparity_from_confusion(confusion, config.fairness_metric)
        for confusion in confusions
    ]
    eo_denominators = global_confusion[:, 1, :].sum(1)
    deo = None
    if bool((eo_denominators > 0).all()):
        global_eo = disparity_from_confusion(global_confusion, "EO")
        deo = abs(float(global_eo[1].item()))
    fr = None if deo is None else 1.0 - deo
    hm = None if fr is None else _harmonic_mean(accuracy, fr)
    local_fairness = [float(disparity.abs().max().item()) for disparity in local_disparities]
    global_fairness = float(selected_global.abs().max().item())
    local_violations = [
        max(0.0, value - config.local_constraint)
        for value in local_fairness
    ]
    return {
        "ACC": accuracy,
        "DEO": deo,
        "SPD": -float(global_dp[0].item()),
        "FR": fr,
        "HM": hm,
        "fairness_metric": config.fairness_metric,
        "global_signed_disparity": [float(value) for value in selected_global.tolist()],
        "global_fairness": global_fairness,
        "local_signed_disparity_by_client": [
            [float(value) for value in disparity.tolist()]
            for disparity in local_disparities
        ],
        "local_fairness_by_client": local_fairness,
        "mean_local_fairness": float(statistics.fmean(local_fairness)),
        "max_local_fairness": max(local_fairness),
        "global_constraint_violation": max(
            0.0, global_fairness - config.global_constraint
        ),
        "mean_local_constraint_violation": float(statistics.fmean(local_violations)),
        "max_local_constraint_violation": max(local_violations),
    }


def evaluate_fedfact(global_model, param_dict, data_bundle, algorithm_state):
    config = FedFACTConfig.from_param_dict(param_dict)
    if len(data_bundle.client_dataset_list) != config.num_clients:
        raise ValueError("FedFACT training dataset count does not match num_clients")
    training_support = build_support_statistics(
        data_bundle.client_dataset_list, config.fairness_metric
    )
    state = validate_fedfact_state(
        algorithm_state,
        global_model,
        config.num_clients,
        config.fairness_metric,
        training_support,
        config.dual_bound,
    )
    test_datasets = data_bundle.client_testing_dataset_list
    test_loaders = data_bundle.client_testing_dataloaders
    if len(test_datasets) != config.num_clients or len(test_loaders) != config.num_clients:
        raise ValueError("FedFACT requires one test dataset and loader per client")
    build_support_statistics(test_datasets, config.fairness_metric)

    device = torch.device(param_dict.get("device", "cpu"))
    theta = global_model.to(device)
    theta.eval()
    confusions = []
    try:
        for client_id in range(config.num_clients):
            phi = copy.deepcopy(global_model).cpu()
            phi.load_state_dict(state["personal_model_states"][client_id], strict=True)
            phi.to(device)
            confusion = _client_confusion(
                theta,
                phi,
                test_loaders[client_id],
                state["ensemble_weights"][client_id],
                device,
            )
            confusions.append(confusion)
            del phi
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        theta.cpu()
    return _metrics_from_confusions(confusions, config)
