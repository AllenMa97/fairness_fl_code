from __future__ import annotations

import copy
import time
from dataclasses import asdict
from typing import Mapping

import torch

from algorithm.client_selection import client_selection
from algorithm.praffl_core import (
    PRAFFL_STATE_SCHEMA_VERSION,
    HyperNetwork,
    PraFFLConfig,
    clone_state_dict_to_cpu,
    uniform_average_state_dicts,
)
from algorithm.praffl_training import train_praffl_client
from tool.amp_utils import get_scaler
from tool.checkpoint import CheckpointState, clean_old_checkpoints, save_checkpoint
from tool.experiment_state import AlgorithmRunResult
from tool.logger import logger
from tool.praffl_evaluation import evaluate_praffl_report


def _validate_model(global_model: torch.nn.Module, task: str) -> tuple[int, int]:
    required = ("bert", "drop", "out", "encode")
    valid = (
        task == "SENT_CLF"
        and all(hasattr(global_model, name) for name in required)
        and isinstance(global_model.out, torch.nn.Linear)
        and global_model.out.bias is not None
        and global_model.out.out_features == 2
    )
    if not valid:
        raise ValueError("PraFFL BERT adaptation requires SENT_CLF with a binary BERT linear head")
    return int(global_model.out.in_features), int(global_model.out.out_features)


def _new_hypernetwork(
    feature_dim: int,
    num_classes: int,
    config: PraFFLConfig,
    repeat_seed: int,
) -> HyperNetwork:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(repeat_seed + config.hypernetwork_seed_offset)
        return HyperNetwork(
            preference_dim=2,
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=config.hypernetwork_hidden_dim,
        )


def _clone_tensor_mapping(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _state_snapshot(
    config: PraFFLConfig,
    template: HyperNetwork,
    client_hypernetworks: Mapping[int, Mapping[str, torch.Tensor]],
    completed_round: int,
    round_metrics_history: list[dict],
) -> dict:
    return {
        "schema_version": PRAFFL_STATE_SCHEMA_VERSION,
        "completed_round": completed_round,
        "round_boundary": True,
        "round_metrics_history": copy.deepcopy(round_metrics_history),
        "config": asdict(config),
        "hypernetwork_spec": {
            "preference_dim": template.preference_dim,
            "feature_dim": template.feature_dim,
            "num_classes": template.num_classes,
            "hidden_dim": template.hidden_dim,
        },
        "client_hypernetworks": {
            int(client_id): _clone_tensor_mapping(state)
            for client_id, state in sorted(client_hypernetworks.items())
        },
    }


def _initial_private_states(template: HyperNetwork, num_clients: int) -> dict[int, dict[str, torch.Tensor]]:
    template_state = clone_state_dict_to_cpu(template)
    return {
        client_id: _clone_tensor_mapping(template_state)
        for client_id in range(num_clients)
    }


def _restore_private_states(
    raw_state: Mapping[str, object],
    template: HyperNetwork,
    config: PraFFLConfig,
    num_clients: int,
    start_round: int,
) -> dict[int, dict[str, torch.Tensor]]:
    if int(raw_state.get("schema_version", -1)) != PRAFFL_STATE_SCHEMA_VERSION:
        raise ValueError("incompatible PraFFL algorithm-state schema")
    if int(raw_state.get("completed_round", -2)) != start_round - 1:
        raise ValueError("PraFFL checkpoint round does not match start_round")
    if raw_state.get("round_boundary") is not True:
        raise ValueError("PraFFL only resumes at a communication-round boundary")
    expected_spec = {
        "preference_dim": template.preference_dim,
        "feature_dim": template.feature_dim,
        "num_classes": template.num_classes,
        "hidden_dim": template.hidden_dim,
    }
    if raw_state.get("hypernetwork_spec") != expected_spec:
        raise ValueError("PraFFL checkpoint hypernetwork shape does not match this run")
    if raw_state.get("config") != asdict(config):
        raise ValueError("PraFFL checkpoint configuration does not match this run")
    raw_private = raw_state.get("client_hypernetworks")
    if not isinstance(raw_private, Mapping) or set(raw_private) != set(range(num_clients)):
        raise ValueError("PraFFL checkpoint must contain exactly one private hypernetwork per client")
    restored = {}
    for client_id in range(num_clients):
        candidate = _clone_tensor_mapping(raw_private[client_id])
        template.load_state_dict(candidate, strict=True)
        restored[client_id] = candidate
    return restored


def _encoder_megabytes(global_model: torch.nn.Module) -> float:
    byte_count = sum(
        tensor.numel() * tensor.element_size()
        for tensor in global_model.bert.state_dict().values()
    )
    return byte_count / (1024 * 1024)


def PraFFL(
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
):
    del testing_dataset_len
    device = torch.device(device)
    feature_dim, num_classes = _validate_model(global_model, param_dict.get("task", ""))
    if len(training_dataloaders) != num_clients_K or len(client_dataset_list) != num_clients_K:
        raise ValueError("PraFFL requires one training loader and dataset entry per client")
    config = PraFFLConfig.from_param_dict(param_dict, algorithm_epoch_T)
    repeat_seed = int(param_dict.get("repeat_seed", param_dict.get("seed", 0)))
    template = _new_hypernetwork(feature_dim, num_classes, config, repeat_seed)
    use_amp = bool(param_dict.get("use_amp", False))
    scaler = get_scaler(device, use_amp)

    if resume_state is None:
        if start_round != 0:
            raise ValueError("nonzero start_round requires a validated CheckpointState")
        private_states = _initial_private_states(template, num_clients_K)
        selection_history: list[list[int]] = []
        round_metrics_history: list[dict] = []
        total_gpu_seconds = 0.0
        total_communication_cost = 0.0
        prior_runtime_seconds = 0.0
    else:
        if resume_state.next_round != start_round:
            raise ValueError("CheckpointState.next_round must equal start_round")
        if resume_state.phase not in {"train", "evaluate"}:
            raise ValueError("PraFFL checkpoint phase must be train or evaluate")
        private_states = _restore_private_states(
            resume_state.algorithm_state,
            template,
            config,
            num_clients_K,
            start_round,
        )
        selection_history = [list(selection) for selection in resume_state.client_selection_history]
        round_metrics_history = copy.deepcopy(
            resume_state.algorithm_state.get("round_metrics_history", [])
        )
        total_gpu_seconds = float(resume_state.total_gpu_seconds)
        total_communication_cost = float(resume_state.total_communication_cost)
        prior_runtime_seconds = float(resume_state.total_runtime_seconds)
        if scaler is not None and resume_state.amp_scaler_state is not None:
            scaler.load_state_dict(resume_state.amp_scaler_state)
        if (scaler is None) != (resume_state.amp_scaler_state is None):
            raise ValueError("PraFFL checkpoint AMP state does not match use_amp")

    if resume_state is not None and resume_state.phase == "evaluate":
        if start_round != communication_round_I:
            raise ValueError("evaluate-phase checkpoint must follow the final training round")
        final_state = _state_snapshot(
            config,
            template,
            private_states,
            start_round - 1,
            round_metrics_history,
        )
        return AlgorithmRunResult(
            global_model=global_model,
            total_gpu_seconds=total_gpu_seconds,
            total_communication_cost=total_communication_cost,
            algorithm_state=final_state,
            amp_scaler_state=None if scaler is None else copy.deepcopy(scaler.state_dict()),
            client_selection_history=selection_history,
        )

    training_dataset_size = len(training_dataset)
    client_sizes = [len(dataset) for dataset in client_dataset_list]
    encoder_mb = _encoder_megabytes(global_model)
    checkpoint_frequency = int(param_dict.get("checkpoint_save_freq", 1))
    keep_latest = 1
    run_started = time.perf_counter()

    logger.info("PraFFL training starts with CPU-resident private hypernetworks")
    for round_index in range(start_round, communication_round_I):
        selected = client_selection(
            client_num=num_clients_K,
            fraction=FL_fraction,
            dataset_size=training_dataset_size,
            client_dataset_size_list=client_sizes,
            drop_rate=FL_drop_rate,
            style="FedAvg",
        )
        selected_ids = [int(client_id) for client_id in selected]
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("PraFFL client selection must be non-empty and unique")
        if any(client_id < 0 or client_id >= num_clients_K for client_id in selected_ids):
            raise ValueError("PraFFL client selection contains an invalid client id")

        uploaded_encoder_states = []
        for client_id in selected_ids:
            result = train_praffl_client(
                global_model,
                template,
                private_states[client_id],
                training_dataloaders[client_id],
                config,
                device,
                use_amp,
                scaler,
            )
            uploaded_encoder_states.append(result.encoder_state)
            private_states[client_id] = _clone_tensor_mapping(result.hypernetwork_state)
            total_gpu_seconds += result.gpu_seconds
        global_model.bert.load_state_dict(
            uniform_average_state_dicts(uploaded_encoder_states),
            strict=True,
        )
        selection_history.append(selected_ids)
        total_communication_cost += 2.0 * encoder_mb * len(selected_ids)
        algorithm_state = _state_snapshot(
            config,
            template,
            private_states,
            round_index,
            round_metrics_history,
        )
        logger.info(
            "PraFFL round %s/%s selected=%s communication_mb=%.6f",
            round_index + 1,
            communication_round_I,
            selected_ids,
            total_communication_cost,
        )

        if round_index + 1 != communication_round_I:
            metrics = evaluate_praffl_report(
                global_model,
                param_dict,
                testing_dataloader,
                algorithm_state,
            )
            round_metrics_history.append({"round": round_index + 1, **metrics})
            algorithm_state = _state_snapshot(
                config,
                template,
                private_states,
                round_index,
                round_metrics_history,
            )
            logger.info(
                "PraFFL round %s evaluation ACC=%.6f DEO=%.6f SPD=%.6f",
                round_index + 1,
                metrics["ACC"],
                metrics["DEO"],
                metrics["SPD"],
            )
            try:
                from tool.tensorboard_logger import flush, log_test_metrics

                log_test_metrics(
                    accuracy=metrics["ACC"],
                    DEO=metrics["DEO"],
                    SPD=metrics["SPD"],
                    step=round_index + 1,
                    gpu_seconds=total_gpu_seconds,
                    communication_cost=total_communication_cost,
                )
                flush()
            except Exception as error:
                logger.warning("PraFFL round TensorBoard logging failed: %s", error)

        should_checkpoint = checkpoint_frequency > 0 and (
            (round_index + 1) % checkpoint_frequency == 0
            or round_index + 1 == communication_round_I
        )
        if should_checkpoint:
            save_checkpoint(
                param_dict,
                round_index,
                global_model,
                algorithm_state=algorithm_state,
                amp_scaler=scaler,
                total_gpu_seconds=total_gpu_seconds,
                total_runtime_seconds=prior_runtime_seconds + time.perf_counter() - run_started,
                total_communication_cost=total_communication_cost,
                client_selection_history=selection_history,
                extra_state={"algorithm": "PraFFL"},
            )
            clean_old_checkpoints(param_dict, keep_latest=keep_latest)

    final_state = _state_snapshot(
        config,
        template,
        private_states,
        communication_round_I - 1,
        round_metrics_history,
    )
    return AlgorithmRunResult(
        global_model=global_model,
        total_gpu_seconds=total_gpu_seconds,
        total_communication_cost=total_communication_cost,
        algorithm_state=final_state,
        amp_scaler_state=None if scaler is None else copy.deepcopy(scaler.state_dict()),
        client_selection_history=selection_history,
    )
