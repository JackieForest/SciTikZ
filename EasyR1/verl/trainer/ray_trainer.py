# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # Cache for Cycle Consistency RL render service (avoid repeated initialization)
        self._cycle_render_service = None
        self._cycle_reward_module = None
        self._cycle_extract_code_fn = None
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # ===== Cycle Consistency RL: Second Generation (Image' -> Code') for Validation =====
            code_prime_responses = None
            rendered_image_paths = None
            
            # Check if Cycle Consistency RL is enabled (same logic as training)
            enable_cycle_consistency = getattr(self.config.worker.reward, 'enable_cycle_consistency', False)
            reward_function_name = getattr(self.config.worker.reward, 'reward_function', '')
            if 'cycle_consistency' in reward_function_name.lower():
                enable_cycle_consistency = True
            
            if enable_cycle_consistency:
                # Step 1: Extract Code from first generation
                # test_output_gen_batch is B*n (sample batch)
                code_responses = []
                if "responses" in test_output_gen_batch.batch:
                    response_ids = test_output_gen_batch.batch["responses"]
                    response_mask = test_output_gen_batch.batch.get("response_mask", None)
                    if response_mask is not None:
                        response_length = torch.sum(response_mask, dim=-1)
                    else:
                        response_length = torch.tensor([response_ids.shape[1]] * response_ids.shape[0])
                    
                    for i in range(len(test_output_gen_batch)):
                        cur_length = int(response_length[i].item())
                        valid_ids = response_ids[i][:cur_length]
                        code_str = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                        code_responses.append(code_str)
                
                # Step 2: Render Code -> Image' (render Code to get Image' for second generation)
                rendered_image_paths = []
                valid_for_stage2 = []  # Track which images are valid for Stage2 (aspect ratio <= 15:1)
                try:
                    if self._cycle_render_service is None:
                        # Initialize render service once (cached, same as training)
                        import importlib.util
                        reward_fn_path = self.config.worker.reward.reward_function
                        if ":" in reward_fn_path:
                            reward_fn_path = reward_fn_path.split(":")[0]
                        
                        if not os.path.isabs(reward_fn_path):
                            possible_paths = [
                                reward_fn_path,
                                os.path.join(os.getcwd(), reward_fn_path),
                                # Try EASYR1_PATH environment variable if set
                                *([os.path.join(easyr1_path, reward_fn_path)] if (easyr1_path := os.environ.get("EASYR1_PATH")) else []),
                            ]
                            for pp in possible_paths:
                                if os.path.exists(pp):
                                    reward_fn_path = pp
                                    break
                        
                        spec = importlib.util.spec_from_file_location("reward_fn_module", reward_fn_path)
                        self._cycle_reward_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(self._cycle_reward_module)
                        
                        if hasattr(self._cycle_reward_module, '_get_services'):
                            self._cycle_render_service, _ = self._cycle_reward_module._get_services(
                                **self.config.worker.reward.reward_function_kwargs
                            )
                            self._cycle_extract_code_fn = getattr(self._cycle_reward_module, 'extract_code_from_response', lambda x: x)
                    
                    # Helper function to validate PNG file before using it (same as training)
                    def _is_valid_image(path: str, min_bytes: int = 1024) -> bool:
                        """Validate that a PNG file exists, has minimum size, and is readable by PIL."""
                        if not path or not isinstance(path, str):
                            return False
                        try:
                            if not os.path.exists(path):
                                return False
                            file_size = os.path.getsize(path)
                            if file_size < min_bytes:
                                return False
                            from PIL import Image
                            with Image.open(path) as im:
                                im.verify()  # Verify file integrity
                            return True
                        except Exception:
                            return False
                    
                    # Render each Code to get Image' and check aspect ratio
                    invalid_image_count = 0
                    for idx, code in enumerate(code_responses):
                        code_clean = self._cycle_extract_code_fn(code) if self._cycle_extract_code_fn else code
                        ok, rendered_png, err_tag = self._cycle_render_service.render(code_clean) if self._cycle_render_service else (False, None, "no_service")
                        
                        # CRITICAL: Validate PNG file before using it
                        # Even if render() returns ok=True, the PNG file might be:
                        # - Empty/truncated (convert failed but path exists)
                        # - Corrupted (not a valid PNG)
                        # - Not yet written (race condition)
                        if ok and rendered_png:
                            if not _is_valid_image(rendered_png):
                                invalid_image_count += 1
                                # Log first few invalid images for debugging
                                if invalid_image_count <= 5:
                                    file_exists = os.path.exists(rendered_png) if rendered_png else False
                                    file_size = os.path.getsize(rendered_png) if file_exists else 0
                                    print(f"[Cycle Consistency RL Validation] WARNING: Invalid PNG file (sample {idx}): path={rendered_png}, exists={file_exists}, size={file_size}, err_tag={err_tag}")
                                rendered_image_paths.append(None)
                                valid_for_stage2.append(False)
                                continue
                            
                            # Check aspect ratio: if > 15:1, skip Stage2 for this sample
                            aspect_ratio_ok = True
                            try:
                                from PIL import Image
                                img = Image.open(rendered_png)
                                width, height = img.size
                                if width > 0 and height > 0:
                                    aspect_ratio = max(width / height, height / width)
                                    if aspect_ratio > 15.0:
                                        aspect_ratio_ok = False
                                        if not hasattr(self, '_val_aspect_ratio_skip_count'):
                                            self._val_aspect_ratio_skip_count = 0
                                        self._val_aspect_ratio_skip_count += 1
                                        if self._val_aspect_ratio_skip_count <= 10:
                                            print(f"[Cycle Consistency RL Validation] Skipping Stage2 for extreme aspect ratio: {width}x{height} (ratio={aspect_ratio:.2f} > 15:1)")
                            except Exception as e:
                                # If we can't check aspect ratio, mark as invalid for Stage2 (fail-safe)
                                if invalid_image_count <= 5:
                                    print(f"[Cycle Consistency RL Validation] WARNING: Failed to check aspect ratio (sample {idx}): {e}")
                                aspect_ratio_ok = False
                            
                            rendered_image_paths.append(rendered_png)
                            valid_for_stage2.append(aspect_ratio_ok)
                        else:
                            rendered_image_paths.append(None)
                            valid_for_stage2.append(False)
                    
                    # Log summary if there were invalid images
                    if invalid_image_count > 0:
                        if not hasattr(self, '_cycle_invalid_image_total_val'):
                            self._cycle_invalid_image_total_val = 0
                        self._cycle_invalid_image_total_val += invalid_image_count
                        print(f"[Cycle Consistency RL Validation] Batch summary: {invalid_image_count} invalid PNG files (total so far: {self._cycle_invalid_image_total_val})")
                except Exception as e:
                    print(f"[Cycle Consistency RL Validation] ERROR: Failed to render codes: {e}")
                    import traceback
                    traceback.print_exc()
                    rendered_image_paths = [None] * len(code_responses)
                    valid_for_stage2 = [False] * len(code_responses)
                
                # Step 3: Generate Code' from Image' (second generation using rendered Image')
                # Only generate Code' for samples with valid aspect ratio (<= 15:1)
                code_prime_responses = None
                # Debug: Log how many valid samples we have for Stage2
                valid_count = sum(1 for p, v in zip(rendered_image_paths, valid_for_stage2) if p and v)
                total_count = len(rendered_image_paths)
                if total_count > 0:
                    print(f"[Cycle Consistency RL Validation] Stage2 readiness: {valid_count}/{total_count} samples have valid Image' for Stage2 generation")
                if rendered_image_paths and any(p and v for p, v in zip(rendered_image_paths, valid_for_stage2)):
                    # CRITICAL: Use actual batch size from test_output_gen_batch to determine repeat_times
                    actual_batch_size_after_gen = len(test_output_gen_batch)
                    original_batch_size = len(test_gen_batch) - pad_size if pad_size > 0 else len(test_gen_batch)
                    actual_repeat_times = actual_batch_size_after_gen // original_batch_size if original_batch_size > 0 else repeat_times
                    
                    # Create batch for second generation using rendered Image'
                    # Start from test_gen_batch (B) which has all required prompt-side fields
                    test_gen_batch_unpadded = unpad_dataproto(test_gen_batch, pad_size=pad_size) if pad_size > 0 else test_gen_batch
                    test_gen_code_prime_batch_base = deepcopy(test_gen_batch_unpadded)
                    
                    # Repeat test_gen_batch from B to B*n first
                    test_gen_code_prime_batch_full = test_gen_code_prime_batch_base.repeat(
                        repeat_times=actual_repeat_times,
                        interleave=True
                    )
                    
                    # Filter: only keep samples with valid aspect ratio (<= 15:1) AND existing image files
                    valid_indices = []
                    for i in range(len(rendered_image_paths)):
                        if (rendered_image_paths[i] and 
                            i < len(valid_for_stage2) and 
                            valid_for_stage2[i] and
                            os.path.exists(rendered_image_paths[i])):
                            valid_indices.append(i)
                        elif rendered_image_paths[i] and not os.path.exists(rendered_image_paths[i]):
                            # Log missing file (limit spam)
                            if not hasattr(self, '_val_missing_image_file_count'):
                                self._val_missing_image_file_count = 0
                            self._val_missing_image_file_count += 1
                            if self._val_missing_image_file_count <= 5:
                                print(f"[Cycle Consistency RL Validation] WARNING: Image file does not exist: {rendered_image_paths[i]}")
                    
                    if valid_indices:
                        # Create filtered batch with only valid samples
                        test_gen_code_prime_batch = test_gen_code_prime_batch_full[valid_indices]
                        
                        # Replace multi_modal_data with rendered Image' paths (only for valid samples)
                        if "multi_modal_data" in test_gen_code_prime_batch.non_tensor_batch:
                            original_multi_modal_data = test_gen_code_prime_batch.non_tensor_batch["multi_modal_data"]
                            new_multi_modal_data = []
                            
                            # Track which valid_indices actually have files (defensive check)
                            actually_valid_indices = []
                            for idx_in_valid, i in enumerate(valid_indices):
                                rendered_path = rendered_image_paths[i]
                                # Double-check file exists before adding to batch (defensive)
                                if not os.path.exists(rendered_path):
                                    if not hasattr(self, '_val_missing_image_file_count'):
                                        self._val_missing_image_file_count = 0
                                    self._val_missing_image_file_count += 1
                                    if self._val_missing_image_file_count <= 5:
                                        print(f"[Cycle Consistency RL Validation] WARNING: Image file disappeared before Stage2: {rendered_path}")
                                    continue  # Skip this sample
                                
                                actually_valid_indices.append(i)
                                if idx_in_valid < len(original_multi_modal_data):
                                    item = original_multi_modal_data[idx_in_valid]
                                    if hasattr(item, 'item'):
                                        item = item.item()
                                    if isinstance(item, dict):
                                        new_item = item.copy()
                                        new_item["images"] = [rendered_path]
                                        new_multi_modal_data.append(new_item)
                                    else:
                                        new_multi_modal_data.append({"images": [rendered_path]})
                                else:
                                    new_multi_modal_data.append({"images": [rendered_path]})
                            
                            # Update valid_indices and batch if some files disappeared
                            if len(actually_valid_indices) < len(valid_indices):
                                valid_indices = actually_valid_indices
                                test_gen_code_prime_batch = test_gen_code_prime_batch_full[valid_indices]
                            
                            test_gen_code_prime_batch.non_tensor_batch["multi_modal_data"] = np.array(new_multi_modal_data, dtype=object)
                        
                        # Force n=1 for stage2 to avoid dimension explosion
                        if not hasattr(test_gen_code_prime_batch, 'meta_info') or test_gen_code_prime_batch.meta_info is None:
                            test_gen_code_prime_batch.meta_info = {}
                        test_gen_code_prime_batch.meta_info["n"] = 1
                        
                        # Generate Code' using rendered Image' (only for valid samples)
                        test_gen_code_prime_batch, pad_size_prime = pad_dataproto_to_divisor(test_gen_code_prime_batch, self.actor_rollout_ref_wg.world_size)
                        test_output_gen_code_prime_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_code_prime_batch)
                        test_output_gen_code_prime_batch = unpad_dataproto(test_output_gen_code_prime_batch, pad_size=pad_size_prime)
                        
                        # Extract Code' responses and map back to full B*n dimension
                        # valid samples get Code', invalid (aspect ratio > 15:1) get empty string
                        code_prime_responses_full = [""] * len(rendered_image_paths)  # Initialize all as empty
                        if "responses" in test_output_gen_code_prime_batch.batch:
                            response_ids_prime = test_output_gen_code_prime_batch.batch["responses"]
                            response_mask_prime = test_output_gen_code_prime_batch.batch.get("response_mask", None)
                            if response_mask_prime is not None:
                                response_length_prime = torch.sum(response_mask_prime, dim=-1)
                            else:
                                response_length_prime = torch.tensor([response_ids_prime.shape[1]] * response_ids_prime.shape[0])
                            
                            # Map Code' responses back to full B*n dimension using valid_indices
                            for output_idx, original_idx in enumerate(valid_indices):
                                if output_idx < len(test_output_gen_code_prime_batch):
                                    cur_length = int(response_length_prime[output_idx].item())
                                    valid_ids = response_ids_prime[output_idx][:cur_length]
                                    code_prime_str = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                    code_prime_responses_full[original_idx] = code_prime_str
                        
                        code_prime_responses = code_prime_responses_full
                        del test_gen_code_prime_batch, test_output_gen_code_prime_batch
                    else:
                        # No valid samples for Stage2 (all have aspect ratio > 15:1 or no valid Image')
                        valid_count = sum(1 for p, v in zip(rendered_image_paths, valid_for_stage2) if p and v)
                        print(f"[Cycle Consistency RL] WARNING: No valid samples for Stage2. Total samples: {len(rendered_image_paths)}, Valid: {valid_count}")
                        code_prime_responses = [""] * len(rendered_image_paths)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            # Ensure multi_modal_data is correctly passed: restore from test_gen_batch
            if "multi_modal_data" in test_gen_batch.non_tensor_batch:
                # Get original multi_modal_data from test_gen_batch
                original_multi_modal_data = test_gen_batch.non_tensor_batch["multi_modal_data"]
                # Repeat multi_modal_data to match the number of generated responses
                repeated_multi_modal_data = np.repeat(original_multi_modal_data, repeat_times, axis=0)
                # Ensure test_output_gen_batch has correct multi_modal_data (override or add)
                test_output_gen_batch.non_tensor_batch["multi_modal_data"] = repeated_multi_modal_data
            test_batch = test_batch.union(test_output_gen_batch)
            
            # ===== Add Code' (second generation) to batch for Cycle Consistency RL (Validation) =====
            if code_prime_responses is not None:
                expected_len = len(test_batch)  # Should be B*n
                if len(code_prime_responses) != expected_len:
                    print(f"[Cycle Consistency RL Validation] WARNING: Dimension mismatch! "
                          f"code_prime_responses length={len(code_prime_responses)}, "
                          f"test_batch length={expected_len}, "
                          f"repeat_times={repeat_times}")
                else:
                    test_batch.non_tensor_batch["code_prime"] = np.array(code_prime_responses, dtype=object)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        # Initialize Cycle Consistency RL diagnostic metrics
        all_metrics['cycle/p_render_ok'] = []
        all_metrics['cycle/p_codeprime_nonempty'] = []
        all_metrics['cycle/len_codeprime_mean'] = []
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch (first generation: Image -> Code)
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            # ===== Cycle Consistency RL: Second Generation (Image' -> Code') =====
            # Correct flow:
            # 1. Extract Code from first generation
            # 2. Render Code -> Image' (rendered image from Code)
            # 3. Generate Code' from Image' (second generation using rendered Image')
            # 4. Compare Code vs Code' for consistency
            code_prime_responses = None
            rendered_image_paths = None
            
            # Check if Cycle Consistency RL is enabled (default: False for safety)
            enable_cycle_consistency = getattr(self.config.worker.reward, 'enable_cycle_consistency', False)
            reward_function_name = getattr(self.config.worker.reward, 'reward_function', '')
            # Only auto-enable if reward function name explicitly contains cycle_consistency
            if 'cycle_consistency' in reward_function_name.lower():
                enable_cycle_consistency = True
            
            if enable_cycle_consistency:
                # Step 1: Extract Code from first generation
                # CRITICAL: gen_batch_output is B*n (sample batch), NOT B!
                # generate_sequences() reads rollout.n config and outputs B*n responses directly.
                # This is why we later do new_batch.repeat(n) to align prompt-side fields.
                code_responses = []
                if "responses" in gen_batch_output.batch:
                    response_ids = gen_batch_output.batch["responses"]
                    response_mask = gen_batch_output.batch.get("response_mask", None)
                    if response_mask is not None:
                        response_length = torch.sum(response_mask, dim=-1)
                    else:
                        response_length = torch.tensor([response_ids.shape[1]] * response_ids.shape[0])
                    
                    # Extract all B*n responses (each sample has its own Code)
                    for i in range(len(gen_batch_output)):
                        cur_length = int(response_length[i].item())
                        valid_ids = response_ids[i][:cur_length]
                        code_str = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                        code_responses.append(code_str)
                
                # Step 2: Render Code -> Image' (render Code to get Image' for second generation)
                # Use cached render service to avoid repeated initialization
                try:
                    if self._cycle_render_service is None:
                        # Initialize render service once (cached)
                        import importlib.util
                        reward_fn_path = self.config.worker.reward.reward_function
                        if ":" in reward_fn_path:
                            reward_fn_path = reward_fn_path.split(":")[0]
                        
                        # Make path absolute if needed
                        if not os.path.isabs(reward_fn_path):
                            possible_paths = [
                                reward_fn_path,
                                os.path.join(os.getcwd(), reward_fn_path),
                                # Try EASYR1_PATH environment variable if set
                                *([os.path.join(easyr1_path, reward_fn_path)] if (easyr1_path := os.environ.get("EASYR1_PATH")) else []),
                            ]
                            for pp in possible_paths:
                                if os.path.exists(pp):
                                    reward_fn_path = pp
                                    break
                        
                        spec = importlib.util.spec_from_file_location("reward_fn_module", reward_fn_path)
                        self._cycle_reward_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(self._cycle_reward_module)
                        
                        if hasattr(self._cycle_reward_module, '_get_services'):
                            self._cycle_render_service, _ = self._cycle_reward_module._get_services(
                                **self.config.worker.reward.reward_function_kwargs
                            )
                            self._cycle_extract_code_fn = getattr(self._cycle_reward_module, 'extract_code_from_response', lambda x: x)
                    
                    # Helper function to validate PNG file before using it
                    def _is_valid_image(path: str, min_bytes: int = 1024) -> bool:
                        """Validate that a PNG file exists, has minimum size, and is readable by PIL."""
                        if not path or not isinstance(path, str):
                            return False
                        try:
                            if not os.path.exists(path):
                                return False
                            file_size = os.path.getsize(path)
                            if file_size < min_bytes:
                                return False
                            from PIL import Image
                            with Image.open(path) as im:
                                im.verify()  # Verify file integrity
                            return True
                        except Exception:
                            return False
                    
                    # Render each Code to get Image' and check aspect ratio
                    # If aspect ratio > 15:1, skip Stage2 (don't generate Code')
                    rendered_image_paths = []
                    valid_for_stage2 = []  # Track which images are valid for Stage2 (aspect ratio <= 15:1)
                    render_ok_count = 0
                    invalid_image_count = 0
                    for idx, code in enumerate(code_responses):
                        code_clean = self._cycle_extract_code_fn(code) if self._cycle_extract_code_fn else code
                        ok, rendered_png, err_tag = self._cycle_render_service.render(code_clean) if self._cycle_render_service else (False, None, "no_service")
                        
                        # CRITICAL: Validate PNG file before using it
                        # Even if render() returns ok=True, the PNG file might be:
                        # - Empty/truncated (convert failed but path exists)
                        # - Corrupted (not a valid PNG)
                        # - Not yet written (race condition)
                        if ok and rendered_png:
                            if not _is_valid_image(rendered_png):
                                invalid_image_count += 1
                                # Log first few invalid images for debugging
                                if invalid_image_count <= 5:
                                    file_exists = os.path.exists(rendered_png) if rendered_png else False
                                    file_size = os.path.getsize(rendered_png) if file_exists else 0
                                    print(f"[Cycle Consistency RL] WARNING: Invalid PNG file (sample {idx}): path={rendered_png}, exists={file_exists}, size={file_size}, err_tag={err_tag}")
                                rendered_image_paths.append(None)
                                valid_for_stage2.append(False)
                                continue
                            
                            # Check aspect ratio: if > 15:1, skip Stage2 for this sample
                            aspect_ratio_ok = True
                            try:
                                from PIL import Image
                                img = Image.open(rendered_png)
                                width, height = img.size
                                if width > 0 and height > 0:
                                    aspect_ratio = max(width / height, height / width)
                                    if aspect_ratio > 15.0:
                                        aspect_ratio_ok = False
                                        if not hasattr(self, '_train_aspect_ratio_skip_count'):
                                            self._train_aspect_ratio_skip_count = 0
                                        self._train_aspect_ratio_skip_count += 1
                                        if self._train_aspect_ratio_skip_count <= 10:
                                            print(f"[Cycle Consistency RL] Skipping Stage2 for extreme aspect ratio: {width}x{height} (ratio={aspect_ratio:.2f} > 15:1)")
                            except Exception as e:
                                # If we can't check aspect ratio, mark as invalid for Stage2 (fail-safe)
                                if invalid_image_count <= 5:
                                    print(f"[Cycle Consistency RL] WARNING: Failed to check aspect ratio (sample {idx}): {e}")
                                aspect_ratio_ok = False
                            
                            rendered_image_paths.append(rendered_png)
                            valid_for_stage2.append(aspect_ratio_ok)
                            render_ok_count += 1
                        else:
                            rendered_image_paths.append(None)
                            valid_for_stage2.append(False)
                    
                    # Log summary if there were invalid images
                    if invalid_image_count > 0:
                        if not hasattr(self, '_cycle_invalid_image_total'):
                            self._cycle_invalid_image_total = 0
                        self._cycle_invalid_image_total += invalid_image_count
                        print(f"[Cycle Consistency RL] Batch summary: {invalid_image_count} invalid PNG files (total so far: {self._cycle_invalid_image_total})")
                    
                    # Store render stats for metrics (avoid print spam)
                    if not hasattr(self, '_cycle_render_stats'):
                        self._cycle_render_stats = {'total': 0, 'ok': 0, 'invalid': 0}
                    self._cycle_render_stats['total'] += len(code_responses)
                    self._cycle_render_stats['ok'] += render_ok_count
                    self._cycle_render_stats['invalid'] += invalid_image_count
                except Exception as e:
                    print(f"[Cycle Consistency RL] ERROR: Failed to render codes: {e}")
                    import traceback
                    traceback.print_exc()
                    rendered_image_paths = [None] * len(code_responses)
                
                # Step 3: Generate Code' from Image' (second generation using rendered Image')
                # CRITICAL DIMENSION ALIGNMENT (FIXED):
                # - gen_batch is B (prompt batch, has all required prompt-side fields)
                # - gen_batch_output is B*n (sample batch, generate_sequences outputs n responses per prompt)
                # - code_responses is B*n (one Code per sample)
                # - rendered_image_paths is B*n (one Image' per sample)
                # - Stage2 MUST run on sample dimension (B*n), but MUST start from gen_batch (B) with all fields
                # - Each sample gets its own code_prime, not shared across n samples
                # Only generate Code' for samples with valid aspect ratio (<= 15:1)
                code_prime_responses = None
                if rendered_image_paths and any(p and v for p, v in zip(rendered_image_paths, valid_for_stage2)):
                    # Create batch for second generation using rendered Image'
                    # CRITICAL: Only include samples with aspect ratio <= 15:1
                    # Start from gen_batch (B) which has all required prompt-side fields
                    gen_code_prime_batch_base = deepcopy(gen_batch)
                    
                    # Repeat gen_batch from B to B*n first
                    gen_code_prime_batch_full = gen_code_prime_batch_base.repeat(
                        repeat_times=self.config.worker.rollout.n, 
                        interleave=True
                    )
                    
                    # Filter: only keep samples with valid aspect ratio (<= 15:1) AND existing image files
                    valid_indices = []
                    for i in range(len(rendered_image_paths)):
                        if (rendered_image_paths[i] and 
                            i < len(valid_for_stage2) and 
                            valid_for_stage2[i] and
                            os.path.exists(rendered_image_paths[i])):
                            valid_indices.append(i)
                        elif rendered_image_paths[i] and not os.path.exists(rendered_image_paths[i]):
                            # Log missing file (limit spam)
                            if not hasattr(self, '_missing_image_file_count'):
                                self._missing_image_file_count = 0
                            self._missing_image_file_count += 1
                            if self._missing_image_file_count <= 5:
                                print(f"[Cycle Consistency RL] WARNING: Image file does not exist: {rendered_image_paths[i]}")
                    
                    if valid_indices:
                        # Create filtered batch with only valid samples
                        gen_code_prime_batch = gen_code_prime_batch_full[valid_indices]
                        
                        # Replace multi_modal_data with rendered Image' paths (only for valid samples)
                        if "multi_modal_data" in gen_code_prime_batch.non_tensor_batch:
                            original_multi_modal_data = gen_code_prime_batch.non_tensor_batch["multi_modal_data"]
                            new_multi_modal_data = []
                            
                            # Track which valid_indices actually have files (defensive check)
                            actually_valid_indices = []
                            for idx_in_valid, i in enumerate(valid_indices):
                                rendered_path = rendered_image_paths[i]
                                # Double-check file exists before adding to batch (defensive)
                                if not os.path.exists(rendered_path):
                                    if not hasattr(self, '_missing_image_file_count'):
                                        self._missing_image_file_count = 0
                                    self._missing_image_file_count += 1
                                    if self._missing_image_file_count <= 5:
                                        print(f"[Cycle Consistency RL] WARNING: Image file disappeared before Stage2: {rendered_path}")
                                    continue  # Skip this sample
                                
                                actually_valid_indices.append(i)
                                if idx_in_valid < len(original_multi_modal_data):
                                    item = original_multi_modal_data[idx_in_valid]
                                    if hasattr(item, 'item'):
                                        item = item.item()
                                    if isinstance(item, dict):
                                        new_item = item.copy()
                                        new_item["images"] = [rendered_path]
                                        new_multi_modal_data.append(new_item)
                                    else:
                                        new_multi_modal_data.append({"images": [rendered_path]})
                                else:
                                    new_multi_modal_data.append({"images": [rendered_path]})
                            
                            # Update valid_indices and batch if some files disappeared
                            if len(actually_valid_indices) < len(valid_indices):
                                valid_indices = actually_valid_indices
                                gen_code_prime_batch = gen_code_prime_batch_full[valid_indices]
                            
                            gen_code_prime_batch.non_tensor_batch["multi_modal_data"] = np.array(new_multi_modal_data, dtype=object)
                    
                        # CRITICAL FIX: Force n=1 for stage2 to avoid (B*n)*n = B*n² explosion
                        if not hasattr(gen_code_prime_batch, 'meta_info') or gen_code_prime_batch.meta_info is None:
                            gen_code_prime_batch.meta_info = {}
                        gen_code_prime_batch.meta_info["n"] = 1
                        
                        # Generate Code' using rendered Image' (only for valid samples)
                        gen_code_prime_batch, pad_size_prime = pad_dataproto_to_divisor(gen_code_prime_batch, self.actor_rollout_ref_wg.world_size)
                        gen_code_prime_output = self.actor_rollout_ref_wg.generate_sequences(gen_code_prime_batch)
                        gen_code_prime_output = unpad_dataproto(gen_code_prime_output, pad_size=pad_size_prime)
                        
                        # Extract Code' responses and map back to full B*n dimension
                        # valid samples get Code', invalid (aspect ratio > 15:1) get empty string
                        code_prime_responses_full = [""] * len(rendered_image_paths)  # Initialize all as empty
                        codeprime_nonempty_count = 0
                        if "responses" in gen_code_prime_output.batch:
                            response_ids_prime = gen_code_prime_output.batch["responses"]
                            response_mask_prime = gen_code_prime_output.batch.get("response_mask", None)
                            if response_mask_prime is not None:
                                response_length_prime = torch.sum(response_mask_prime, dim=-1)
                            else:
                                response_length_prime = torch.tensor([response_ids_prime.shape[1]] * response_ids_prime.shape[0])
                            
                            # Map Code' responses back to full B*n dimension using valid_indices
                            for output_idx, original_idx in enumerate(valid_indices):
                                if output_idx < len(gen_code_prime_output):
                                    cur_length = int(response_length_prime[output_idx].item())
                                    valid_ids = response_ids_prime[output_idx][:cur_length]
                                    code_prime_str = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                    code_prime_responses_full[original_idx] = code_prime_str
                                    if code_prime_str.strip():
                                        codeprime_nonempty_count += 1
                        
                        # Store stats for metrics
                        if not hasattr(self, '_cycle_codeprime_stats'):
                            self._cycle_codeprime_stats = {'total': 0, 'nonempty': 0}
                        self._cycle_codeprime_stats['total'] += len(rendered_image_paths)
                        self._cycle_codeprime_stats['nonempty'] += codeprime_nonempty_count
                        
                        code_prime_responses = code_prime_responses_full
                        del gen_code_prime_batch, gen_code_prime_output
                    else:
                        # No valid samples for Stage2 (all have aspect ratio > 15:1)
                        code_prime_responses = [""] * len(rendered_image_paths)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            # Ensure multi_modal_data is correctly passed: restore from gen_batch (generation process may not preserve it)
            if "multi_modal_data" in gen_batch.non_tensor_batch:
                # Get original multi_modal_data from gen_batch
                original_multi_modal_data = gen_batch.non_tensor_batch["multi_modal_data"]
                # Repeat multi_modal_data to match the number of generated responses (n responses)
                repeated_multi_modal_data = np.repeat(original_multi_modal_data, self.config.worker.rollout.n, axis=0)
                # Ensure gen_batch_output has correct multi_modal_data (override or add)
                gen_batch_output.non_tensor_batch["multi_modal_data"] = repeated_multi_modal_data
            new_batch = new_batch.union(gen_batch_output)
            
            # ===== Add Code' (second generation) to batch for Cycle Consistency RL =====
            # CRITICAL (FIXED): code_prime_responses is B*n (one per sample), NOT B!
            # No need to repeat - each sample already has its own code_prime
            if code_prime_responses is not None:
                # Verify dimension alignment
                expected_len = len(new_batch)  # Should be B*n
                if len(code_prime_responses) != expected_len:
                    print(f"[Cycle Consistency RL] WARNING: Dimension mismatch! "
                          f"code_prime_responses length={len(code_prime_responses)}, "
                          f"new_batch length={expected_len}, "
                          f"rollout.n={self.config.worker.rollout.n}")
                    # Truncate or pad to match
                    if len(code_prime_responses) > expected_len:
                        code_prime_responses = code_prime_responses[:expected_len]
                    else:
                        code_prime_responses.extend([""] * (expected_len - len(code_prime_responses)))
                
                # Add Code' to non_tensor_batch (already B*n, one per sample)
                new_batch.non_tensor_batch["code_prime"] = np.array(code_prime_responses, dtype=object)
                
                # Add diagnostic metrics for Cycle Consistency RL
                if hasattr(self, '_cycle_render_stats') and self._cycle_render_stats['total'] > 0:
                    p_render_ok = self._cycle_render_stats['ok'] / self._cycle_render_stats['total']
                    all_metrics['cycle/p_render_ok'].append(p_render_ok)
                
                if hasattr(self, '_cycle_codeprime_stats') and self._cycle_codeprime_stats['total'] > 0:
                    p_codeprime_nonempty = self._cycle_codeprime_stats['nonempty'] / self._cycle_codeprime_stats['total']
                    all_metrics['cycle/p_codeprime_nonempty'].append(p_codeprime_nonempty)
                    # Average length of non-empty code_prime (from current batch)
                    if code_prime_responses:
                        nonempty_responses = [cp for cp in code_prime_responses if cp.strip()]
                        avg_len = sum(len(cp) for cp in nonempty_responses) / max(1, len(nonempty_responses))
                        all_metrics['cycle/len_codeprime_mean'].append(avg_len)
                
                # Log summary every N steps (avoid spam)
                if not hasattr(self, '_cycle_log_counter'):
                    self._cycle_log_counter = 0
                self._cycle_log_counter += 1
                if self._cycle_log_counter % 10 == 0:  # Log every 10 batches
                    render_msg = ""
                    if hasattr(self, '_cycle_render_stats'):
                        render_msg = f"render_ok={self._cycle_render_stats['ok']}/{self._cycle_render_stats['total']}"
                    codeprime_msg = ""
                    if hasattr(self, '_cycle_codeprime_stats'):
                        codeprime_msg = f"codeprime_nonempty={self._cycle_codeprime_stats['nonempty']}/{self._cycle_codeprime_stats['total']}"
                    if render_msg or codeprime_msg:
                        print(f"[Cycle Consistency RL] Summary (batch {self._cycle_log_counter}): {render_msg} {codeprime_msg}")

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    raise RuntimeError("No sample is kept after filtering. Please check your data.")

                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                # Always log cycle metrics (not just when online_filtering=True)
                # Extract cycle metrics separately to ensure they're always logged
                cycle_metrics = {}
                for key in ['cycle/p_render_ok', 'cycle/p_codeprime_nonempty', 'cycle/len_codeprime_mean']:
                    if key in all_metrics and len(all_metrics[key]) > 0:
                        cycle_metrics[f"reward/{key}"] = np.mean(all_metrics[key])
                
                if cycle_metrics:
                    metrics.update(cycle_metrics)
                
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
