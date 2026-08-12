from __future__ import annotations

import copy
import hashlib
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from jongbench.dataset import DEFAULT_PTS, GameplayIterable, discover_logs
from jongbench.evaluate import load_checkpoint, networks_from_checkpoint, resolve_device
from jongbench.mortal_model import AuxNet, ConfidenceHead, PolicyHead


@dataclass
class TrainConfig:
    logs: str
    init: str
    out: str
    steps: int = 4000
    batch_size: int = 256
    device: str = "auto"
    lr: float = 3e-4
    encoder_lr_scale: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    teacher_temperature: float = 0.1
    policy_ce_weight: float = 1.0
    policy_kl_weight: float = 0.1
    q_distill_weight: float = 1.0
    rank_weight: float = 0.2
    confidence_weight: float = 0.05
    freeze_encoder: bool = True
    file_batch_size: int = 4
    validation_ratio: float = 0.1
    validation_batches: int = 20
    log_every: int = 50
    save_every: int = 1000
    pts: tuple[float, float, float, float] = DEFAULT_PTS
    data_provenance: str | None = None
    data_sha256: str | None = None
    seed: int = 20260812


@dataclass
class PolicyRLConfig:
    logs: str
    init: str
    out: str
    anchor: str | None = None
    steps: int = 128
    batch_size: int = 512
    device: str = "auto"
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    clip_ratio: float = 0.2
    target_kl: float | None = 0.03
    anchor_kl_weight: float = 0.02
    entropy_weight: float = 0.001
    sampling_temperature: float = 1.0
    file_batch_size: int = 8
    log_every: int = 10
    pts: tuple[float, float, float, float] = DEFAULT_PTS
    duplicate_challenger_only: bool = False
    seed: int = 20260812


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.requires_grad_(trainable)
    if not trainable:
        module.eval()


def _move_batch(batch, device: torch.device):
    obs, actions, masks, steps_to_done, rewards, ranks = batch
    return (
        obs.to(device, dtype=torch.float32, non_blocking=True),
        actions.to(device, dtype=torch.int64, non_blocking=True),
        masks.to(device, dtype=torch.bool, non_blocking=True),
        steps_to_done.to(device, dtype=torch.int64, non_blocking=True),
        rewards.to(device, dtype=torch.float32, non_blocking=True),
        ranks.to(device, dtype=torch.int64, non_blocking=True),
    )


def split_train_validation(
    files: list[str], validation_ratio: float
) -> tuple[list[str], list[str]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if validation_ratio == 0 or len(files) < 2:
        return list(files), []
    validation_count = max(1, round(len(files) * validation_ratio))
    validation_count = min(validation_count, len(files) - 1)
    ordered = sorted(
        files,
        key=lambda path: hashlib.sha256(Path(path).name.encode("utf-8")).digest(),
    )
    validation_files = ordered[:validation_count]
    train_files = ordered[validation_count:]
    return train_files, validation_files


def _legal_kl(
    log_policy: torch.Tensor,
    teacher_policy: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    teacher = teacher_policy[mask]
    student_log = log_policy[mask]
    divergence = teacher * (teacher.clamp_min(1e-12).log() - student_log)
    return divergence.sum() / log_policy.shape[0]


def _optimizer_groups(
    modules: list[nn.Module],
    *,
    lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    parameters = [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    decay = [parameter for parameter in parameters if parameter.ndim >= 2]
    no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


@torch.inference_mode()
def evaluate_policy(
    *,
    brain: nn.Module,
    dqn: nn.Module,
    policy: nn.Module,
    aux_net: nn.Module | None,
    confidence: nn.Module | None,
    files: list[str],
    version: int,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    teacher_temperature: float,
    file_batch_size: int,
) -> dict[str, float]:
    if not files or max_batches <= 0:
        return {}
    modules = [brain, dqn, policy]
    if confidence is not None:
        modules.append(confidence)
    if aux_net is not None:
        modules.append(aux_net)
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()

    totals = {
        "count": 0.0,
        "top1": 0.0,
        "top3": 0.0,
        "nll": 0.0,
        "teacher_top1": 0.0,
        "teacher_nll": 0.0,
        "teacher_kl": 0.0,
        "confidence": 0.0,
        "confidence_brier": 0.0,
        "rank_top1": 0.0,
        "rank_nll": 0.0,
    }
    dataset = GameplayIterable(
        files,
        version=version,
        file_batch_size=file_batch_size,
        infinite=False,
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            obs, actions, masks, _, _, ranks = _move_batch(batch, device)
            contested = masks.sum(-1) > 1
            if not bool(contested.any()):
                continue
            obs = obs[contested]
            actions = actions[contested]
            masks = masks[contested]
            ranks = ranks[contested]
            phi = brain(obs)
            teacher_logits = dqn(phi, masks) / teacher_temperature
            logits = policy(phi, masks)
            count = actions.shape[0]
            top3 = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
            log_pi = F.log_softmax(logits, dim=-1)
            teacher_pi = F.softmax(teacher_logits, dim=-1)
            totals["count"] += count
            totals["top1"] += (logits.argmax(-1) == actions).sum().item()
            totals["top3"] += top3.eq(actions.unsqueeze(-1)).any(-1).sum().item()
            totals["nll"] += F.cross_entropy(logits, actions, reduction="sum").item()
            totals["teacher_top1"] += (
                (teacher_logits.argmax(-1) == actions).sum().item()
            )
            totals["teacher_nll"] += F.cross_entropy(
                teacher_logits, actions, reduction="sum"
            ).item()
            totals["teacher_kl"] += (
                _legal_kl(log_pi, teacher_pi, masks) * count
            ).item()
            if confidence is not None:
                predicted_confidence = confidence(phi)
                correct = (logits.argmax(-1) == actions).float()
                totals["confidence"] += predicted_confidence.sum().item()
                totals["confidence_brier"] += F.mse_loss(
                    predicted_confidence, correct, reduction="sum"
                ).item()
            if aux_net is not None:
                (rank_logits,) = aux_net(phi)
                totals["rank_top1"] += (rank_logits.argmax(-1) == ranks).sum().item()
                totals["rank_nll"] += F.cross_entropy(
                    rank_logits, ranks, reduction="sum"
                ).item()
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.train(mode)

    count = totals.pop("count")
    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def train(cfg: TrainConfig) -> dict[str, Any]:
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    if cfg.teacher_temperature <= 0:
        raise ValueError("teacher_temperature must be positive")
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    files = discover_logs(cfg.logs)
    if not files:
        raise ValueError(f"no gameplay logs under {cfg.logs}")
    train_files, validation_files = split_train_validation(files, cfg.validation_ratio)

    ckpt = load_checkpoint(cfg.init, map_location="cpu")
    brain, dqn, policy, aux_net, confidence, version = networks_from_checkpoint(ckpt)
    if version != 4:
        raise ValueError(f"reviewer training requires Mortal version 4, got {version}")
    if policy is None:
        policy = PolicyHead.from_dqn(dqn, cfg.teacher_temperature)
    if aux_net is None:
        aux_net = AuxNet((4,))
    if confidence is None:
        confidence = ConfidenceHead()

    brain = brain.to(device)
    dqn = dqn.to(device)
    policy = policy.to(device)
    aux_net = aux_net.to(device)
    confidence = confidence.to(device)
    teacher_brain = copy.deepcopy(brain).eval() if not cfg.freeze_encoder else brain
    for param in teacher_brain.parameters():
        param.requires_grad = False
    _set_trainable(brain, not cfg.freeze_encoder)
    _set_trainable(dqn, False)
    brain.freeze_bn(True)
    policy.train()
    aux_net.train()
    confidence.train()
    if not cfg.freeze_encoder:
        brain.train()

    parameter_groups = _optimizer_groups(
        [policy, aux_net, confidence],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    if not cfg.freeze_encoder:
        parameter_groups.extend(
            _optimizer_groups(
                [brain],
                lr=cfg.lr * cfg.encoder_lr_scale,
                weight_decay=cfg.weight_decay,
            )
        )
    trainable = [
        parameter for group in parameter_groups for parameter in group["params"]
    ]
    optimizer = torch.optim.AdamW(parameter_groups)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    baseline_metrics = evaluate_policy(
        brain=brain,
        dqn=dqn,
        policy=policy,
        confidence=confidence,
        aux_net=aux_net,
        files=validation_files,
        version=version,
        device=device,
        batch_size=cfg.batch_size,
        max_batches=cfg.validation_batches,
        teacher_temperature=cfg.teacher_temperature,
        file_batch_size=cfg.file_batch_size,
    )
    if baseline_metrics:
        print(
            "validation baseline "
            + " ".join(f"{key}={value:.4f}" for key, value in baseline_metrics.items())
        )

    dataset = GameplayIterable(
        train_files,
        version=version,
        pts=cfg.pts,
        file_batch_size=cfg.file_batch_size,
        infinite=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batches = iter(loader)
    running: dict[str, float] = {}
    running_count = 0
    last_stats: dict[str, Any] = {}
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        obs, actions, masks, _, _, ranks = _move_batch(next(batches), device)
        contested = masks.sum(-1) > 1
        if not bool(contested.any()):
            continue
        obs = obs[contested]
        actions = actions[contested]
        masks = masks[contested]
        ranks = ranks[contested]
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            with torch.no_grad():
                teacher_phi = teacher_brain(obs)
                teacher_q = dqn(teacher_phi, masks)
                teacher_pi = F.softmax(teacher_q / cfg.teacher_temperature, dim=-1)

            phi = brain(obs) if not cfg.freeze_encoder else teacher_phi
            policy_logits = policy(phi, masks)
            log_pi = F.log_softmax(policy_logits, dim=-1)
            policy_ce = ce(policy_logits, actions)
            policy_kl = _legal_kl(log_pi, teacher_pi, masks)

            if cfg.freeze_encoder:
                q_distill = torch.zeros((), device=device)
            else:
                q_out = dqn(phi, masks)
                q_distill = (q_out[masks] - teacher_q[masks]).square().mean()

            (rank_logits,) = aux_net(phi)
            rank_loss = ce(rank_logits, ranks)
            correct = (policy_logits.argmax(-1) == actions).detach().float()
            with torch.autocast(device.type, enabled=False):
                conf = confidence(phi.float()).clamp(1e-6, 1 - 1e-6)
                conf_loss = F.binary_cross_entropy(conf, correct)
            loss = (
                cfg.policy_ce_weight * policy_ce
                + cfg.policy_kl_weight * policy_kl
                + cfg.q_distill_weight * q_distill
                + cfg.rank_weight * rank_loss
                + cfg.confidence_weight * conf_loss
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable, cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            human_top1 = float(
                (policy_logits.argmax(-1) == actions).float().mean().detach()
            )
        stats = {
            "loss": float(loss.detach()),
            "policy_ce": float(policy_ce.detach()),
            "policy_kl": float(policy_kl.detach()),
            "human_top1": human_top1,
            "q_distill": float(q_distill.detach()),
            "rank": float(rank_loss.detach()),
            "confidence": float(conf_loss.detach()),
        }
        for key, value in stats.items():
            running[key] = running.get(key, 0.0) + value
        running_count += 1

        if step % cfg.log_every == 0 or step == 1 or step == cfg.steps:
            last_stats = {key: value / running_count for key, value in running.items()}
            elapsed = time.time() - t0
            print(
                f"step {step}/{cfg.steps} "
                + " ".join(f"{key}={value:.4f}" for key, value in last_stats.items())
                + f" elapsed={elapsed:.1f}s device={device}"
            )
            running = {}
            running_count = 0

        if step % cfg.save_every == 0 and step != cfg.steps:
            _save_checkpoint(
                cfg.out,
                ckpt,
                brain=brain,
                dqn=dqn,
                policy=policy,
                aux_net=aux_net,
                confidence=confidence,
                steps=step,
                train_cfg=cfg,
                train_files=len(train_files),
                validation_files=len(validation_files),
                metrics={"baseline": baseline_metrics},
            )
            print(f"saved {cfg.out}")

    validation_metrics = evaluate_policy(
        brain=brain,
        dqn=dqn,
        policy=policy,
        confidence=confidence,
        aux_net=aux_net,
        files=validation_files,
        version=version,
        device=device,
        batch_size=cfg.batch_size,
        max_batches=cfg.validation_batches,
        teacher_temperature=cfg.teacher_temperature,
        file_batch_size=cfg.file_batch_size,
    )
    if validation_metrics:
        print(
            "validation trained "
            + " ".join(
                f"{key}={value:.4f}" for key, value in validation_metrics.items()
            )
        )
    _save_checkpoint(
        cfg.out,
        ckpt,
        brain=brain,
        dqn=dqn,
        policy=policy,
        aux_net=aux_net,
        confidence=confidence,
        steps=cfg.steps,
        train_cfg=cfg,
        train_files=len(train_files),
        validation_files=len(validation_files),
        metrics={
            "baseline": baseline_metrics,
            "validation": validation_metrics,
        },
    )
    print(f"saved {cfg.out}")
    last_stats["baseline"] = baseline_metrics
    last_stats["validation"] = validation_metrics
    return last_stats


def train_policy_rl(cfg: PolicyRLConfig) -> dict[str, float]:
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    if cfg.sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be positive")
    if not 0 < cfg.clip_ratio < 1:
        raise ValueError("clip_ratio must be in (0, 1)")
    if cfg.target_kl is not None and cfg.target_kl <= 0:
        raise ValueError("target_kl must be positive or None")
    if cfg.anchor_kl_weight < 0 or cfg.entropy_weight < 0:
        raise ValueError("regularization weights must be non-negative")

    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    files = discover_logs(cfg.logs)
    if not files:
        raise ValueError(f"no gameplay logs under {cfg.logs}")

    ckpt = load_checkpoint(cfg.init, map_location="cpu")
    brain, dqn, policy, _, _, version = networks_from_checkpoint(ckpt)
    if version != 4:
        raise ValueError(f"policy RL requires Mortal version 4, got {version}")
    if policy is None:
        policy = PolicyHead.from_dqn(dqn)

    anchor_path = cfg.anchor or cfg.init
    anchor_ckpt = load_checkpoint(anchor_path, map_location="cpu")
    anchor_brain, anchor_dqn, anchor_policy, _, _, anchor_version = (
        networks_from_checkpoint(anchor_ckpt)
    )
    if anchor_version != version:
        raise ValueError(
            f"anchor Mortal version {anchor_version} does not match policy version {version}"
        )
    if anchor_policy is None:
        anchor_policy = PolicyHead.from_dqn(anchor_dqn)

    shared_anchor_encoder = all(
        torch.equal(value, anchor_brain.state_dict()[key])
        for key, value in brain.state_dict().items()
    )
    brain = brain.to(device).eval()
    brain.requires_grad_(False)
    policy = policy.to(device).train()
    old_policy = copy.deepcopy(policy).eval()
    old_policy.requires_grad_(False)
    anchor_policy = anchor_policy.to(device).eval()
    anchor_policy.requires_grad_(False)
    if shared_anchor_encoder:
        anchor_brain = brain
    else:
        anchor_brain = anchor_brain.to(device).eval()
        anchor_brain.requires_grad_(False)

    parameter_groups = _optimizer_groups(
        [policy],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    trainable = [
        parameter for group in parameter_groups for parameter in group["params"]
    ]
    optimizer = torch.optim.AdamW(parameter_groups)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    reward_values = torch.tensor(cfg.pts, dtype=torch.float32, device=device)
    reward_mean = reward_values.mean()
    reward_std = reward_values.std(correction=0)
    if float(reward_std) == 0:
        raise ValueError("pts must assign at least two distinct placement rewards")

    dataset = GameplayIterable(
        files,
        version=version,
        pts=cfg.pts,
        file_batch_size=cfg.file_batch_size,
        infinite=True,
        duplicate_challenger_only=cfg.duplicate_challenger_only,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batches = iter(loader)
    running: dict[str, float] = {}
    running_count = 0
    last_stats: dict[str, float] = {}
    updates = 0
    t0 = time.time()

    while updates < cfg.steps:
        obs, actions, masks, _, rewards, _ = _move_batch(next(batches), device)
        contested = masks.sum(-1) > 1
        if not bool(contested.any()):
            continue
        obs = obs[contested]
        actions = actions[contested]
        masks = masks[contested]
        rewards = rewards[contested]
        advantages = (rewards - reward_mean) / reward_std

        with (
            torch.no_grad(),
            torch.autocast(device.type, enabled=device.type == "cuda"),
        ):
            phi = brain(obs)
            old_logits = old_policy(phi, masks) / cfg.sampling_temperature
            old_log_policy = F.log_softmax(old_logits, dim=-1)
            old_action_logp = old_log_policy.gather(-1, actions.unsqueeze(-1)).squeeze(
                -1
            )
            anchor_phi = phi if shared_anchor_encoder else anchor_brain(obs)
            anchor_logits = anchor_policy(anchor_phi, masks) / cfg.sampling_temperature
            anchor_pi = F.softmax(anchor_logits, dim=-1)

        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = policy(phi, masks) / cfg.sampling_temperature
            log_policy = F.log_softmax(logits, dim=-1)
            action_logp = log_policy.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            log_ratio = action_logp - old_action_logp
            ratio = log_ratio.exp()
            clipped_ratio = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
            policy_loss = -torch.minimum(
                ratio * advantages, clipped_ratio * advantages
            ).mean()
            anchor_kl = _legal_kl(log_policy, anchor_pi, masks)
            legal_log_policy = log_policy.masked_fill(~masks, 0.0)
            legal_policy = log_policy.exp().masked_fill(~masks, 0.0)
            entropy = -(legal_policy * legal_log_policy).sum(-1).mean()
            loss = (
                policy_loss
                + cfg.anchor_kl_weight * anchor_kl
                - cfg.entropy_weight * entropy
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite policy RL loss at update {updates + 1}"
            )

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable, cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        updates += 1

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
        stats = {
            "loss": float(loss.detach()),
            "policy": float(policy_loss.detach()),
            "anchor_kl": float(anchor_kl.detach()),
            "entropy": float(entropy.detach()),
            "behavior_kl": float(approx_kl.detach()),
            "clip_fraction": float(clip_fraction.detach()),
            "mean_return": float(rewards.mean()),
        }
        for key, value in stats.items():
            running[key] = running.get(key, 0.0) + value
        running_count += 1

        if updates % cfg.log_every == 0 or updates == 1 or updates == cfg.steps:
            last_stats = {key: value / running_count for key, value in running.items()}
            elapsed = time.time() - t0
            print(
                f"policy-rl {updates}/{cfg.steps} "
                + " ".join(f"{key}={value:.4f}" for key, value in last_stats.items())
                + f" elapsed={elapsed:.1f}s device={device}"
            )
            running = {}
            running_count = 0

        if cfg.target_kl is not None and float(approx_kl) > cfg.target_kl:
            print(
                f"policy-rl stopped at {updates}: "
                f"behavior_kl={float(approx_kl):.4f} > {cfg.target_kl:.4f}"
            )
            break

    if running_count:
        last_stats = {key: value / running_count for key, value in running.items()}
    last_stats["updates"] = float(updates)
    _save_policy_rl_checkpoint(
        cfg.out,
        ckpt,
        policy=policy,
        train_cfg=cfg,
        files=len(files),
        updates=updates,
        metrics=last_stats,
    )
    print(f"saved {cfg.out}")
    return last_stats


def _save_policy_rl_checkpoint(
    path: str | Path,
    init_ckpt: dict[str, Any],
    *,
    policy: nn.Module,
    train_cfg: PolicyRLConfig,
    files: int,
    updates: int,
    metrics: dict[str, float],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(init_ckpt.get("config", {}))
    reviewer = config.setdefault("reviewer", {})
    reviewer["has_policy"] = True
    prior = reviewer.get("policy_rl")
    history = list(prior.get("history", [])) if isinstance(prior, dict) else []
    history.append(
        {
            "source_checkpoint": train_cfg.init,
            "anchor_checkpoint": train_cfg.anchor or train_cfg.init,
            "files": files,
            "updates": updates,
            "training": asdict(train_cfg),
            "metrics": dict(metrics),
        }
    )
    reviewer["policy_rl"] = {
        "format": 1,
        "total_updates": sum(int(item.get("updates", 0)) for item in history),
        "history": history,
    }
    payload = dict(init_ckpt)
    payload.update(
        {
            "policy": {
                key: value.detach().cpu() for key, value in policy.state_dict().items()
            },
            "config": config,
            "timestamp": time.time(),
        }
    )
    torch.save(payload, out)


def _save_checkpoint(
    path: str | Path,
    init_ckpt: dict[str, Any],
    *,
    brain,
    dqn,
    policy,
    aux_net,
    confidence,
    steps: int,
    train_cfg: TrainConfig,
    train_files: int,
    validation_files: int,
    metrics: dict[str, Any],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(init_ckpt.get("config", {}))
    config.setdefault("reviewer", {})
    config["reviewer"].update(
        {
            "format": 1,
            "has_policy": True,
            "has_aux": True,
            "has_confidence": True,
            "steps": steps,
            "source_checkpoint": train_cfg.init,
            "data_provenance": train_cfg.data_provenance,
            "train_files": train_files,
            "data_sha256": train_cfg.data_sha256,
            "validation_files": validation_files,
            "training": asdict(train_cfg),
            "metrics": metrics,
        }
    )
    payload = {
        key: value
        for key, value in init_ckpt.items()
        if key
        not in {
            "mortal",
            "current_dqn",
            "policy",
            "aux_net",
            "confidence",
            "config",
            "steps",
            "timestamp",
        }
    }
    payload.update(
        {
            "mortal": {
                key: value.detach().cpu() for key, value in brain.state_dict().items()
            },
            "current_dqn": {
                key: value.detach().cpu() for key, value in dqn.state_dict().items()
            },
            "policy": {
                key: value.detach().cpu() for key, value in policy.state_dict().items()
            },
            "aux_net": {
                key: value.detach().cpu() for key, value in aux_net.state_dict().items()
            },
            "confidence": {
                key: value.detach().cpu()
                for key, value in confidence.state_dict().items()
            },
            "config": config,
            "steps": steps,
            "timestamp": time.time(),
        }
    )
    torch.save(payload, out)
