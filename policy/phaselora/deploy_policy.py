# ----------------------------------------------------------------------------
# PhaseLoRA adapter on official ACT for RoboSynChallenge.
# Loads the released ACT checkpoint, injects P/E-conditioned LoRA, then loads
# the small adapter. Official ACT weights are not overwritten.
# ----------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from policy.act.deploy_policy import (
    _prepare_policy_batch,
    _unnormalize,
    get_model as get_act_model,
    reset_model as reset_act_model,
)
from policy.inference_timing import finish_inference, start_inference
from policy.phaselora.layers import inject_phaselora, set_phase_route
from policy.phaselora.router import HISTORY, PhaseRouter


def _attach_phaselora(policy, adapter_dir: Path, device) -> None:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank = int(config["rank"])
    alpha = float(config["alpha"])
    inject_phaselora(policy.model, rank, alpha)
    visual_dim = int(config["visual_dim"])
    policy.phase_router = PhaseRouter(visual_dim=visual_dim)
    policy.to(device)
    for param in policy.parameters():
        param.requires_grad = False
    weights = load_file(str(adapter_dir / "adapter.safetensors"))
    current = policy.state_dict()
    missing = [key for key in weights if key not in current]
    if missing:
        raise RuntimeError(f"adapter keys not in policy: {missing[:8]}")
    current.update(weights)
    policy.load_state_dict(current, strict=False)
    policy.phase_history_len = int(config.get("history", HISTORY))
    policy.phase_visual_dim = visual_dim
    policy.phase_history = None


def get_model(usr_args):
    adapter_dir = Path(usr_args["checkpoint_path"])
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    act_args = dict(usr_args)
    act_args["checkpoint_path"] = config["base_checkpoint"]
    act_args["temporal_ensemble_coeff"] = None
    act_args["pace"] = False
    act_args["ocrh_stitch"] = False
    act_args["scale_gripper"] = False
    policy = get_act_model(act_args)
    _attach_phaselora(policy, adapter_dir, policy.act_device)
    policy.grasp_hold_phase = bool(usr_args.get("grasp_hold_phase", False))
    policy.eval()
    return policy


def _visual_summary(model, batch) -> torch.Tensor:
    features = []
    for image_key in model.act_image_keys:
        feature_map = model.model.backbone(batch[image_key])["feature_map"]
        features.append(feature_map.mean(dim=(2, 3)))
    visual = torch.cat(features, dim=-1)
    if visual.shape[-1] != model.phase_visual_dim:
        raise RuntimeError(
            f"visual dim {visual.shape[-1]} != router {model.phase_visual_dim}"
        )
    return visual


def _ensure_history(model, batch_size: int, action_dim: int) -> torch.Tensor:
    if model.phase_history is None or model.phase_history.shape[0] != batch_size:
        model.phase_history = torch.zeros(
            batch_size,
            model.phase_history_len,
            action_dim,
            device=model.act_device,
            dtype=torch.float32,
        )
    return model.phase_history


def _left_gripper_open(model, obs) -> bool:
    """Left gripper is index 6. Dataset scale is about 1 when open and 0 when closed."""
    state = obs
    for key in str(model.state_obs_path).split("/"):
        if key:
            state = state[key]
    if isinstance(state, torch.Tensor):
        value = float(state.reshape(-1)[6].detach().float().cpu())
    else:
        value = float(np.asarray(state).reshape(-1)[6])
    return value > 0.5


def _set_route_from_obs(model, batch, obs) -> None:
    if getattr(model, "grasp_hold_phase", False) and _left_gripper_open(model, obs):
        ones = torch.ones(1, device=model.act_device)
        zeros = torch.zeros(1, device=model.act_device)
        set_phase_route(ones, zeros)
        return
    visual = _visual_summary(model, batch)
    history = _ensure_history(model, visual.shape[0], 14)
    precision, event = model.phase_router(visual, batch["observation.state"], history)
    set_phase_route(precision, event)


def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.act_step):
        runs_model_inference = (
            model.config.temporal_ensemble_coeff is not None
            or len(model._action_queue) == 0
        )
        started_at = start_inference(model.act_device) if runs_model_inference else None
        batch = _prepare_policy_batch(model, final_obs)
        if runs_model_inference:
            _set_route_from_obs(model, batch, final_obs)
        action = model.select_action(batch)
        action_mean = model.act_norm_stats.get("unnormalize_outputs.buffer_action.mean")
        action_std = model.act_norm_stats.get("unnormalize_outputs.buffer_action.std")
        if action_mean is not None and action_std is not None:
            action = _unnormalize(action, action_mean, action_std)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action shape [B, D], but got {tuple(action.shape)}.")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = f"Policy action has dim {policy_action_dim}, but env expects {env_action_dim}."
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        action_tensor = action.detach().to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        if runs_model_inference:
            finish_inference(started_at, inference_times_s, model.act_device)
        history = _ensure_history(model, action_tensor.shape[0], action_tensor.shape[-1])
        model.phase_history = torch.cat(
            [history[:, 1:], action_tensor.detach().to(device=model.act_device).unsqueeze(1)],
            dim=1,
        )
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            break
        if isinstance(truncated, torch.Tensor):
            is_truncated = truncated.any().item()
        elif isinstance(truncated, np.ndarray):
            is_truncated = truncated.any()
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    reset_act_model(model)
    model.phase_history = None
