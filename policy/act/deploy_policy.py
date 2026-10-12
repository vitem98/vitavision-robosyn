# ----------------------------------------------------------------------------
# LeRobot ACT Policy Adapter for RoboSynChallenge
#
# Follows the same unified evaluation interface as policy/pi0:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import os

import numpy as np
import torch
from safetensors import safe_open

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from policy.inference_timing import finish_inference, start_inference

_NORM_EPS = 1e-8


def get_model(usr_args):
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = [f"--device={device}"]
    try:
        policy = ACTPolicy.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
        policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()
    _enable_temporal_ensemble(policy, usr_args.get("temporal_ensemble_coeff", 0.01))

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(
            image_key,
            image_key.removeprefix("observation.images."),
        )

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 8))
    policy.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map
    policy.act_norm_stats = _load_norm_stats(checkpoint_path, policy.act_device)
    policy.scale_gripper = bool(usr_args.get("scale_gripper", False))
    policy.pace = bool(usr_args.get("pace", False))
    policy.pace_queue = []
    policy.pace_min_horizon = int(usr_args.get("pace_min_horizon", 8))
    policy.pace_max_horizon = int(usr_args.get("pace_max_horizon", 40))
    policy.ocrh_stitch = bool(usr_args.get("ocrh_stitch", False))
    policy.stitch_lambda = float(usr_args.get("stitch_lambda", 0.35))
    policy.ocrh_horizon = int(usr_args.get("ocrh_horizon", 0) or 0)
    policy.ocrh_queue = []
    policy.ocrh_tail = None
    policy.regrasp_on_miss = bool(usr_args.get("regrasp_on_miss", False))
    policy.regrasp_move_m = float(usr_args.get("regrasp_move_m", 0.05))
    policy.regrasp_grace = int(usr_args.get("regrasp_grace", 15))
    policy.regrasp_cooldown = int(usr_args.get("regrasp_cooldown", 50))
    policy.regrasp_gripper_index = int(usr_args.get("regrasp_gripper_index", 6))
    policy.regrasp_open_threshold = float(usr_args.get("regrasp_open_threshold", 0.5))
    _reset_regrasp_state(policy)

    if policy.act_step <= 0:
        raise ValueError(f"act_step must be positive, got {policy.act_step}.")

    replan_steps = usr_args.get("replan_steps")
    if replan_steps not in (None, "None", "", 0, "0"):
        from collections import deque

        replan_steps = int(replan_steps)
        if replan_steps < 1 or replan_steps > int(policy.config.chunk_size):
            raise ValueError(
                f"replan_steps must be in [1, {policy.config.chunk_size}], got {replan_steps}."
            )
        policy.config.n_action_steps = replan_steps
        policy._action_queue = deque([], maxlen=replan_steps)

    return policy


def _enable_temporal_ensemble(policy, coeff) -> None:
    """Query ACT every step and blend overlapping chunks.

    This is Algorithm 2 from the ACT paper. The released checkpoints leave it
    off and execute a fixed action queue, so small errors on long-horizon tasks
    are never corrected.
    """
    if coeff is None or coeff == "None":
        return
    coeff = float(coeff)
    policy.config.temporal_ensemble_coeff = coeff
    policy.config.n_action_steps = 1
    policy.temporal_ensembler = ACTTemporalEnsembler(coeff, policy.config.chunk_size)


def _load_norm_stats(checkpoint_path, device) -> dict[str, torch.Tensor]:
    """Load the mean/std buffers shipped inside the released ACT weights.

    The installed LeRobot no longer applies these modules, so the raw
    observations were being fed to a network trained on normalized inputs.
    """
    model_file = os.path.join(checkpoint_path, "model.safetensors")
    stats = {}
    if os.path.isfile(model_file):
        with safe_open(model_file, framework="pt", device=str(device)) as handle:
            for key in handle.keys():
                if key.startswith("normalize_inputs.buffer_") or key.startswith(
                    "unnormalize_outputs.buffer_"
                ):
                    stats[key] = handle.get_tensor(key)
    if stats:
        return stats
    return _load_processor_norm_stats(checkpoint_path, device)


def _processor_feature_name(key: str) -> str:
    """Map migrated keys like observation.images.cam.high back to cam_high."""
    renamed = {
        "observation.images.cam.high": "observation.images.cam_high",
        "observation.images.cam.left.wrist": "observation.images.cam_left_wrist",
        "observation.images.cam.right.wrist": "observation.images.cam_right_wrist",
    }
    for source, target in renamed.items():
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def _load_processor_norm_stats(checkpoint_path, device) -> dict[str, torch.Tensor]:
    stats = {}
    pre_file = os.path.join(
        checkpoint_path, "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    post_file = os.path.join(
        checkpoint_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    if os.path.isfile(pre_file):
        with safe_open(pre_file, framework="pt", device=str(device)) as handle:
            for key in handle.keys():
                feature, stat = _processor_feature_name(key).rsplit(".", 1)
                if feature == "action":
                    continue
                buffer = "normalize_inputs.buffer_" + feature.replace(".", "_")
                stats[f"{buffer}.{stat}"] = handle.get_tensor(key)
    if os.path.isfile(post_file):
        with safe_open(post_file, framework="pt", device=str(device)) as handle:
            for key in ("action.mean", "action.std"):
                if key in handle.keys():
                    stat = key.rsplit(".", 1)[1]
                    stats[f"unnormalize_outputs.buffer_action.{stat}"] = handle.get_tensor(key)
    return stats


def _normalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=tensor.device, dtype=tensor.dtype)
    std = std.to(device=tensor.device, dtype=tensor.dtype)
    while mean.ndim < tensor.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (tensor - mean) / (std + _NORM_EPS)


def _unnormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=tensor.device, dtype=tensor.dtype)
    std = std.to(device=tensor.device, dtype=tensor.dtype)
    while mean.ndim < tensor.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return tensor * std + mean


def _prepare_policy_batch(model, obs) -> dict[str, torch.Tensor]:
    state = obs
    for key in str(model.state_obs_path).split("/"):
        if key:
            state = state[key]
    state_tensor = (
        state.detach().to(device=model.act_device, dtype=torch.float32)
        if isinstance(state, torch.Tensor)
        else torch.as_tensor(state, dtype=torch.float32, device=model.act_device)
    )
    if state_tensor.ndim == 1:
        state_tensor = state_tensor.unsqueeze(0)
    state_mean = model.act_norm_stats.get("normalize_inputs.buffer_observation_state.mean")
    state_std = model.act_norm_stats.get("normalize_inputs.buffer_observation_state.std")
    if state_mean is not None and state_std is not None:
        state_tensor = _normalize(state_tensor, state_mean, state_std)

    batch = {"observation.state": state_tensor}
    for image_key in model.act_image_keys:
        camera_name = model.image_key_map.get(
            image_key,
            image_key.removeprefix("observation.images."),
        )
        image = obs["sensor"][camera_name]["color"]
        image_tensor = (
            image.detach().to(device=model.act_device, dtype=torch.float32)
            if isinstance(image, torch.Tensor)
            else torch.as_tensor(image, dtype=torch.float32, device=model.act_device)
        )
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
        if torch.max(image_tensor) > 1.5:
            image_tensor = image_tensor / 255.0
        stat_name = image_key.replace(".", "_")
        image_mean = model.act_norm_stats.get(f"normalize_inputs.buffer_{stat_name}.mean")
        image_std = model.act_norm_stats.get(f"normalize_inputs.buffer_{stat_name}.std")
        if image_mean is not None and image_std is not None:
            image_tensor = _normalize(image_tensor, image_mean, image_std)
        batch[image_key] = image_tensor
    return batch


def _pace_horizon(actions: torch.Tensor, min_h: int, max_h: int) -> int:
    """Execute up to the next slow point in the predicted chunk.

    Arm-joint speed dips at contact and handover. Those are the steps where
    an open-loop chunk should be cut so the policy can look again.
    """
    length = int(actions.shape[0])
    max_h = min(max_h, length)
    min_h = min(min_h, max_h)
    arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    arm = [index for index in arm if index < actions.shape[-1]]
    if length < 3 or not arm:
        return max_h
    speed = (actions[1:, arm] - actions[:-1, arm]).norm(dim=-1)
    minima = torch.where((speed[1:-1] <= speed[:-2]) & (speed[1:-1] <= speed[2:]))[0] + 1
    if minima.numel() == 0:
        return max_h
    threshold = torch.quantile(speed, 0.4)
    for index in minima.tolist():
        horizon = index + 1
        if min_h <= horizon <= max_h and speed[index] <= threshold:
            return horizon
    for index in minima.tolist():
        horizon = index + 1
        if min_h <= horizon <= max_h:
            return horizon
    return max_h


def _fill_pace_queue(model, obs) -> None:
    batch = _prepare_policy_batch(model, obs)
    chunk = model.predict_action_chunk(batch)
    action_mean = model.act_norm_stats.get("unnormalize_outputs.buffer_action.mean")
    action_std = model.act_norm_stats.get("unnormalize_outputs.buffer_action.std")
    if action_mean is not None and action_std is not None:
        chunk = _unnormalize(chunk, action_mean, action_std)
    horizon = _pace_horizon(chunk[0], model.pace_min_horizon, model.pace_max_horizon)
    model.pace_queue = [chunk[0, index] for index in range(horizon)]


def _qpos_vector(model, obs) -> np.ndarray:
    state = obs
    for key in str(model.state_obs_path).split("/"):
        if key:
            state = state[key]
    if isinstance(state, torch.Tensor):
        return state.detach().float().reshape(-1).cpu().numpy()
    return np.asarray(state, dtype=np.float32).reshape(-1)


def _duck_xy(env) -> np.ndarray:
    base = getattr(env, "unwrapped", env)
    pose = base.sim.get_rigid_object("duck").get_local_pose(to_matrix=True)
    xy = pose[0, :2, 3]
    if isinstance(xy, torch.Tensor):
        xy = xy.detach().float().cpu().numpy()
    return np.asarray(xy, dtype=np.float32).reshape(-1)[:2]


def _reset_regrasp_state(model) -> None:
    model._regrasp_origin = None
    model._regrasp_close_step = None
    model._regrasp_steps = 0
    model._regrasp_cooldown = 0
    model._regrasp_closed_once = False


def _flush_regrasp_queue(model, env, obs) -> None:
    """Drop a stale open-loop tail before the next action is chosen."""
    if not getattr(model, "regrasp_on_miss", False):
        return
    if model._regrasp_origin is None:
        model._regrasp_origin = _duck_xy(env)
    if model._regrasp_cooldown > 0 or model._regrasp_closed_once:
        return
    if model._regrasp_close_step is None:
        return
    if model._regrasp_steps < model._regrasp_close_step + int(model.regrasp_grace):
        return
    if len(model._action_queue) == 0:
        return
    qpos = _qpos_vector(model, obs)
    gripper_index = int(model.regrasp_gripper_index)
    if float(qpos[gripper_index]) <= float(model.regrasp_open_threshold):
        model._regrasp_closed_once = True
        return
    moved = float(np.linalg.norm(_duck_xy(env) - model._regrasp_origin))
    if moved < float(model.regrasp_move_m):
        return
    model._action_queue.clear()
    model._regrasp_close_step = None
    model._regrasp_cooldown = int(model.regrasp_cooldown)


def _update_regrasp(model, env, obs, action) -> None:
    """Remember a close command, then protect the following retry chunk.

    Training chunks are 50 steps, and the demos do not close the left gripper
    until step 115. Replanning every few steps never finishes that close.
    A retry is armed only after this chunk commands a close and that close
    has had a short grace period. Once the gripper has actually shut, later
    actions are left alone so the duck can be carried to the drawer.
    """
    if not getattr(model, "regrasp_on_miss", False):
        return
    gripper_index = int(model.regrasp_gripper_index)
    qpos = _qpos_vector(model, obs)
    if float(qpos[gripper_index]) <= float(model.regrasp_open_threshold):
        model._regrasp_closed_once = True
    gripper_command = float(action.reshape(-1)[gripper_index].detach().float().cpu())
    if (
        gripper_command < float(model.regrasp_open_threshold)
        and model._regrasp_close_step is None
        and model._regrasp_cooldown <= 0
    ):
        model._regrasp_close_step = int(model._regrasp_steps)
    if model._regrasp_cooldown > 0:
        model._regrasp_cooldown -= 1
    model._regrasp_steps += 1


def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.act_step):
        _flush_regrasp_queue(model, env, final_obs)
        runs_model_inference = False
        if model.pace:
            if not model.pace_queue:
                runs_model_inference = True
                started_at = start_inference(model.act_device)
                _fill_pace_queue(model, final_obs)
                finish_inference(started_at, inference_times_s, model.act_device)
            action = model.pace_queue.pop(0)
            if action.ndim == 1:
                action = action.unsqueeze(0)
        elif getattr(model, "ocrh_stitch", False):
            if not model.ocrh_queue:
                runs_model_inference = True
                started_at = start_inference(model.act_device)
                batch = _prepare_policy_batch(model, final_obs)
                chunk = model.predict_action_chunk(batch)
                action_mean = model.act_norm_stats.get("unnormalize_outputs.buffer_action.mean")
                action_std = model.act_norm_stats.get("unnormalize_outputs.buffer_action.std")
                if action_mean is not None and action_std is not None:
                    chunk = _unnormalize(chunk, action_mean, action_std)
                if model.ocrh_tail is not None:
                    overlap = min(model.ocrh_tail.shape[0], chunk.shape[1])
                    blend = float(model.stitch_lambda)
                    chunk = chunk.clone()
                    chunk[0, :overlap] = (1.0 - blend) * chunk[0, :overlap] + blend * model.ocrh_tail[:overlap]
                horizon = int(model.ocrh_horizon or model.config.n_action_steps)
                horizon = max(1, min(horizon, int(chunk.shape[1])))
                model.ocrh_queue = [chunk[0, index] for index in range(horizon)]
                model.ocrh_tail = chunk[0, horizon:].detach()
                finish_inference(started_at, inference_times_s, model.act_device)
            action = model.ocrh_queue.pop(0)
            if action.ndim == 1:
                action = action.unsqueeze(0)
        else:
            runs_model_inference = (
                model.config.temporal_ensemble_coeff is not None
                or len(model._action_queue) == 0
            )
            started_at = start_inference(model.act_device) if runs_model_inference else None
            batch = _prepare_policy_batch(model, final_obs)
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
        if getattr(model, "scale_gripper", False):
            for gripper_index in (6, 13):
                if gripper_index < action_tensor.shape[-1]:
                    column = action_tensor[:, gripper_index]
                    action_tensor[:, gripper_index] = column.clamp(0, 1) * 0.05
        _update_regrasp(model, env, final_obs, action_tensor)
        if runs_model_inference and not model.pace and not getattr(model, "ocrh_stitch", False):
            finish_inference(started_at, inference_times_s, model.act_device)
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
    model.reset()
    if getattr(model, "pace_queue", None) is not None:
        model.pace_queue.clear()
    if getattr(model, "ocrh_queue", None) is not None:
        model.ocrh_queue.clear()
    model.ocrh_tail = None
    if getattr(model, "regrasp_on_miss", False):
        _reset_regrasp_state(model)
