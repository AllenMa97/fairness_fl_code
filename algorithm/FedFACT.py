# FedFACT-In: A Provable Framework for Controllable Group-Fairness Calibration
# Paper: https://arxiv.org/abs/2506.03777
# Official implementation: https://github.com/liizhang/FedFACT

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Mapping

import torch

from algorithm.Optimizers import BERTCLF_Optimizer
from algorithm.fedfact_core import (
    FedFACTConfig,
    SupportError,
    build_calibration_matrices,
    build_support_statistics,
    calibrated_loss,
    confusion_from_predictions,
    disparity_from_confusion,
    ensemble_probabilities,
    initialize_fedfact_state,
    update_dual,
    update_ensemble_weight,
    validate_fedfact_state,
)
from tool.amp_utils import autocast_context, get_scaler
from tool.checkpoint import CheckpointState, clean_old_checkpoints, save_checkpoint
from tool.experiment_state import AlgorithmRunResult
from tool.logger import logger


@dataclass(frozen=True)
class ClientAudit:
    theta_loss: float
    phi_loss: float
    confusion: torch.Tensor
    sample_count: int


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _deep_cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _deep_cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _forward_text_logits(model, batch, device):
    result = model(
        input_ids=batch["input_ids"].to(device, non_blocking=True),
        attention_mask=batch["attention_mask"].to(device, non_blocking=True),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("FedFACT SENT_CLF model must return (features, logits)")
    logits = result[1]
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("FedFACT requires a two-logit classifier")
    return logits


def _audit_current_ensemble(theta, phi, loader, matrices, weight, device, use_amp):
    theta.eval()
    phi.eval()
    theta_loss_sum = phi_loss_sum = 0.0
    sample_count = 0
    predictions, labels, protected = [], [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["labels"].to(device, non_blocking=True).reshape(-1).long()
            a = batch["protected"].to(device, non_blocking=True).reshape(-1).long()
            with autocast_context(device, use_amp):
                theta_logits = _forward_text_logits(theta, batch, device)
                phi_logits = _forward_text_logits(phi, batch, device)
                theta_loss = calibrated_loss(theta_logits, y, a, matrices)
                phi_loss = calibrated_loss(phi_logits, y, a, matrices)
                probabilities = ensemble_probabilities(theta_logits, phi_logits, weight)
            n = y.numel()
            theta_loss_sum += theta_loss.item() * n
            phi_loss_sum += phi_loss.item() * n
            sample_count += n
            predictions.append(probabilities.argmax(1).cpu())
            labels.append(y.cpu())
            protected.append(a.cpu())
    if sample_count == 0:
        raise SupportError("FedFACT client loader is empty")
    return ClientAudit(
        theta_loss_sum / sample_count,
        phi_loss_sum / sample_count,
        confusion_from_predictions(
            torch.cat(predictions), torch.cat(labels), torch.cat(protected)
        ),
        sample_count,
    )


def _make_optimizer(model, param_dict):
    optimizer = BERTCLF_Optimizer(
        method=str(param_dict["optimize_method"]).lower(),
        learning_rate=float(param_dict["learning_rate"]),
        max_grad_norm=0,
    )
    optimizer.set_parameters(list(model.named_parameters()))
    return optimizer


def _step_both(theta_optimizer, phi_optimizer, scaler):
    if scaler is None:
        theta_optimizer.step()
        phi_optimizer.step()
    else:
        scaler.step(theta_optimizer)
        scaler.step(phi_optimizer)
        scaler.update()


def _train_theta_and_phi(
    theta,
    phi,
    loader,
    matrices,
    epochs,
    param_dict,
    device,
    use_amp,
    scaler,
    batch_trace=None,
):
    theta.train()
    phi.train()
    theta_optimizer = _make_optimizer(theta, param_dict)
    phi_optimizer = _make_optimizer(phi, param_dict)
    for epoch in range(epochs):
        theta_loss_sum = phi_loss_sum = 0.0
        sample_count = 0
        for batch in loader:
            if batch_trace is not None:
                batch_trace.append(batch["input_ids"].reshape(-1).tolist())
            theta_optimizer.zero_grad()
            phi_optimizer.zero_grad()
            y = batch["labels"].to(device, non_blocking=True).reshape(-1).long()
            a = batch["protected"].to(device, non_blocking=True).reshape(-1).long()
            with autocast_context(device, use_amp):
                theta_loss = calibrated_loss(
                    _forward_text_logits(theta, batch, device), y, a, matrices
                )
                phi_loss = calibrated_loss(
                    _forward_text_logits(phi, batch, device), y, a, matrices
                )
            if scaler is None:
                theta_loss.backward()
                phi_loss.backward()
            else:
                scaler.scale(theta_loss).backward()
                scaler.scale(phi_loss).backward()
            _step_both(theta_optimizer, phi_optimizer, scaler)
            n = y.numel()
            theta_loss_sum += theta_loss.detach().item() * n
            phi_loss_sum += phi_loss.detach().item() * n
            sample_count += n
        if sample_count == 0:
            raise SupportError("FedFACT client loader is empty")
        logger.info(
            "FedFACT local epoch %s: theta_loss=%.8f phi_loss=%.8f",
            epoch + 1,
            theta_loss_sum / sample_count,
            phi_loss_sum / sample_count,
        )


class StreamingModelAverage:
    def __init__(self, reference_state: Mapping[str, torch.Tensor], total_weight: float):
        if total_weight <= 0:
            raise ValueError("FedFACT aggregation total weight must be positive")
        self.reference = {name: value.detach().cpu().clone() for name, value in reference_state.items()}
        self.total_weight = float(total_weight)
        self.added_weight = 0.0
        self.accumulator = {
            name: torch.zeros_like(value, dtype=torch.float64)
            for name, value in self.reference.items()
            if value.is_floating_point()
        }

    def add(self, state_dict: Mapping[str, torch.Tensor], sample_weight: float):
        if set(state_dict) != set(self.reference) or sample_weight <= 0:
            raise ValueError("invalid unified state for FedFACT aggregation")
        coefficient = float(sample_weight) / self.total_weight
        for name, reference in self.reference.items():
            value = state_dict[name].detach().cpu()
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"FedFACT aggregation mismatch for {name}")
            if reference.is_floating_point():
                self.accumulator[name].add_(value.to(torch.float64), alpha=coefficient)
            elif not torch.equal(value, reference):
                raise ValueError(f"FedFACT non-floating buffer differs for {name}")
        self.added_weight += float(sample_weight)

    def finish(self) -> dict[str, torch.Tensor]:
        if abs(self.added_weight - self.total_weight) > max(1e-9, self.total_weight * 1e-12):
            raise ValueError("FedFACT aggregation did not receive the full client weight")
        result = {}
        for name, reference in self.reference.items():
            if reference.is_floating_point():
                result[name] = self.accumulator[name].to(reference.dtype)
            else:
                result[name] = reference.clone()
        return result


def _model_megabytes(model: torch.nn.Module) -> float:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    ) / (1024 * 1024)


def _validate_model(model: torch.nn.Module):
    if not hasattr(model, "out") or not isinstance(model.out, torch.nn.Linear) or model.out.out_features != 2:
        raise ValueError("FedFACT requires BERT plus a binary linear classification head")


def FedFACT(
    device,
    global_model,
    algorithm_epoch_T,
    num_clients_K,
    communication_round_I,
    FL_fraction,
    FL_drop_rate,
    training_dataloaders,
    training_dataset,
    client_dataset_list,
    param_dict,
    testing_dataloader,
    testing_dataset_len,
    start_round=0,
    resume_state: CheckpointState | None = None,
    data_bundle=None,
):
    del training_dataset, testing_dataloader, testing_dataset_len
    effective = dict(param_dict)
    duplicates = {
        "num_clients_K": num_clients_K,
        "FL_fraction": FL_fraction,
        "FL_drop_rate": FL_drop_rate,
        "communication_round_I": communication_round_I,
        "algorithm_epoch_T": algorithm_epoch_T,
    }
    for name, value in duplicates.items():
        if name in effective and float(effective[name]) != float(value):
            raise ValueError(f"FedFACT positional {name} conflicts with param_dict")
        effective[name] = value
    config = FedFACTConfig.from_param_dict(effective)
    if len(training_dataloaders) != num_clients_K or len(client_dataset_list) != num_clients_K:
        raise ValueError("FedFACT requires one training loader and dataset per client")

    # Fail closed before model copies, device transfers, scalers, or optimizers.
    support = build_support_statistics(client_dataset_list, config.fairness_metric)
    _validate_model(global_model)
    device = torch.device(device)
    use_amp = bool(effective.get("use_amp", False))
    scaler = get_scaler(device, use_amp)

    if resume_state is None:
        if start_round != 0:
            raise ValueError("nonzero start_round requires a validated CheckpointState")
        state = initialize_fedfact_state(
            global_model,
            num_clients_K,
            config.fairness_metric,
            support,
            config.dual_init,
            config.ensemble_weight_init,
        )
        history = []
        total_gpu_seconds = 0.0
        total_communication_cost = 0.0
        prior_runtime_seconds = 0.0
    else:
        if start_round not in {0, resume_state.next_round}:
            raise ValueError("FedFACT start_round conflicts with checkpoint")
        start_round = resume_state.next_round
        if resume_state.phase not in {"train", "evaluate"}:
            raise ValueError("FedFACT checkpoint phase must be train or evaluate")
        state = validate_fedfact_state(
            resume_state.algorithm_state,
            global_model,
            num_clients_K,
            config.fairness_metric,
            support,
            config.dual_bound,
        )
        history = [list(ids) for ids in resume_state.client_selection_history]
        total_gpu_seconds = float(resume_state.total_gpu_seconds)
        total_communication_cost = float(resume_state.total_communication_cost)
        prior_runtime_seconds = float(resume_state.total_runtime_seconds)
        if scaler is not None and resume_state.amp_scaler_state is not None:
            scaler.load_state_dict(resume_state.amp_scaler_state)
        if (scaler is None) != (resume_state.amp_scaler_state is None):
            raise ValueError("FedFACT checkpoint AMP state does not match use_amp")
        if resume_state.phase == "evaluate":
            return AlgorithmRunResult(
                global_model=global_model.cpu(),
                total_gpu_seconds=total_gpu_seconds,
                total_communication_cost=total_communication_cost,
                algorithm_state=_deep_cpu_clone(state),
                amp_scaler_state=None if scaler is None else copy.deepcopy(scaler.state_dict()),
                client_selection_history=history,
            )

    global_model.cpu()
    model_mb = _model_megabytes(global_model)
    checkpoint_frequency = int(effective.get("checkpoint_save_freq", 1))
    run_started = time.monotonic()

    for iter_t in range(start_round, communication_round_I):
        round_confusions = []
        next_personal_states = []
        next_local_duals = state["local_duals"].clone()
        next_weights = state["ensemble_weights"].clone()
        aggregator = StreamingModelAverage(
            global_model.state_dict(),
            total_weight=float(support.client_totals.sum().item()),
        )
        for client_id in range(num_clients_K):
            matrices = build_calibration_matrices(
                client_id,
                support.counts,
                config.fairness_metric,
                state["global_dual"],
                state["local_duals"][client_id],
                config.calibration_epsilon,
            )
            theta = copy.deepcopy(global_model).to(device)
            phi = copy.deepcopy(global_model)
            phi.load_state_dict(state["personal_model_states"][client_id], strict=True)
            phi.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            client_started = time.perf_counter()
            audit = _audit_current_ensemble(
                theta,
                phi,
                training_dataloaders[client_id],
                matrices,
                state["ensemble_weights"][client_id],
                device,
                use_amp,
            )
            round_confusions.append(audit.confusion)
            next_weights[client_id] = update_ensemble_weight(
                state["ensemble_weights"][client_id],
                audit.theta_loss,
                audit.phi_loss,
                config.ensemble_learning_rate,
            )
            local_disparity = disparity_from_confusion(audit.confusion, config.fairness_metric)
            next_local_duals[client_id] = update_dual(
                state["local_duals"][client_id],
                local_disparity,
                config.local_constraint,
                config.dual_learning_rate,
                config.dual_bound,
            )
            _train_theta_and_phi(
                theta,
                phi,
                training_dataloaders[client_id],
                matrices,
                algorithm_epoch_T,
                effective,
                device,
                use_amp,
                scaler,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                total_gpu_seconds += time.perf_counter() - client_started
            next_personal_states.append(_cpu_state_dict(phi))
            aggregator.add(_cpu_state_dict(theta), support.client_totals[client_id].item())
            del theta, phi
            if device.type == "cuda":
                torch.cuda.empty_cache()

        global_confusion = torch.stack(round_confusions).sum(dim=0)
        global_disparity = disparity_from_confusion(global_confusion, config.fairness_metric)
        state["global_dual"] = update_dual(
            state["global_dual"],
            global_disparity,
            config.global_constraint,
            config.dual_learning_rate,
            config.dual_bound,
        )
        state["local_duals"] = next_local_duals
        state["ensemble_weights"] = next_weights
        state["personal_model_states"] = next_personal_states
        global_model.load_state_dict(aggregator.finish(), strict=True)
        history.append(list(range(num_clients_K)))
        total_communication_cost += 2 * num_clients_K * model_mb
        logger.info(
            "FedFACT round %s/%s global_disparity=%s communication_mb=%.6f",
            iter_t + 1,
            communication_round_I,
            global_disparity.tolist(),
            total_communication_cost,
        )
        if iter_t + 1 < communication_round_I:
            if data_bundle is None:
                raise ValueError(
                    "FedFACT requires FederatedDataBundle for per-round personalized evaluation"
                )
            from algorithm.fedfact_evaluation import evaluate_fedfact

            round_metrics = evaluate_fedfact(
                global_model, effective, data_bundle, state
            )
            state["round_metrics_history"].append(
                {"round": iter_t + 1, **copy.deepcopy(round_metrics)}
            )
            logger.info(
                "FedFACT round %s evaluation ACC=%.6f DEO=%s SPD=%.6f",
                iter_t + 1,
                round_metrics["ACC"],
                round_metrics["DEO"],
                round_metrics["SPD"],
            )
            try:
                from tool.tensorboard_logger import flush, log_test_metrics
                log_test_metrics(
                    accuracy=round_metrics["ACC"],
                    DEO=round_metrics["DEO"],
                    SPD=round_metrics["SPD"],
                    step=iter_t + 1,
                    gpu_seconds=total_gpu_seconds,
                    communication_cost=total_communication_cost,
                )
                flush()
            except Exception as error:
                logger.warning("FedFACT round TensorBoard logging failed: %s", error)
        if checkpoint_frequency > 0 and (
            (iter_t + 1) % checkpoint_frequency == 0
            or iter_t + 1 == communication_round_I
        ):
            save_checkpoint(
                param_dict,
                iter_t,
                global_model,
                algorithm_state=_deep_cpu_clone(state),
                amp_scaler=scaler,
                total_gpu_seconds=total_gpu_seconds,
                total_runtime_seconds=prior_runtime_seconds + time.monotonic() - run_started,
                total_communication_cost=total_communication_cost,
                client_selection_history=history,
            )
            clean_old_checkpoints(param_dict, keep_latest=1)

    return AlgorithmRunResult(
        global_model=global_model.cpu(),
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state=_deep_cpu_clone(state),
        amp_scaler_state=None if scaler is None else copy.deepcopy(scaler.state_dict()),
        client_selection_history=history,
    )
