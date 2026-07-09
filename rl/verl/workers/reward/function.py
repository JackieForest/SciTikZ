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

import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict


import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict, total=False):
    response: str
    response_length: int
    ground_truth: str
    image_path: str  # Optional: image path (for visual consistency reward)
    code_prime: str  # Optional: second generated code (for Cycle Consistency RL)


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": cur_response_length,
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        
        # Try to get image paths (for visual consistency reward)
        image_paths = None
        if "multi_modal_data" in data.non_tensor_batch:
            multi_modal_data = data.non_tensor_batch["multi_modal_data"]
            image_paths = []
            for i in range(len(data)):
                # Support list, tuple, numpy.ndarray
                if i < len(multi_modal_data):
                    item = multi_modal_data[i]
                    # Handle elements in numpy array (may be numpy object)
                    if hasattr(item, 'item'):
                        item = item.item()
                    
                    if isinstance(item, dict) and "images" in item:
                        images = item["images"]
                        # Handle images in numpy array
                        if hasattr(images, 'tolist'):
                            images = images.tolist()
                        if isinstance(images, (list, tuple)) and len(images) > 0:
                            # Extract first image path (support absolute and relative paths)
                            image_path = images[0] if isinstance(images[0], str) else str(images[0])
                            image_paths.append(image_path)
                        else:
                            image_paths.append(None)
                    else:
                        image_paths.append(None)
                else:
                    image_paths.append(None)
        else:
            print(f"[reward_function] WARNING: multi_modal_data not found in non_tensor_batch. Available keys: {list(data.non_tensor_batch.keys())}")
        
        # Try to get Code' (second generated code, for Cycle Consistency RL)
        code_prime_list = None
        if "code_prime" in data.non_tensor_batch:
            code_prime_list = data.non_tensor_batch["code_prime"]
        elif "response_prime" in data.non_tensor_batch:
            code_prime_list = data.non_tensor_batch["response_prime"]
        
        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            
            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["ground_truth"][i],
            }
            
            # If image path exists, add to reward_input (for visual consistency reward)
            # Improved image_path extraction: support multiple formats (images, image, image_path)
            if image_paths is not None and i < len(image_paths) and image_paths[i] is not None:
                reward_input["image_path"] = image_paths[i]
            elif "multi_modal_data" in data.non_tensor_batch and i < len(data.non_tensor_batch["multi_modal_data"]):
                # Fallback: try to extract from multi_modal_data directly
                item = data.non_tensor_batch["multi_modal_data"][i]
                if hasattr(item, 'item'):
                    item = item.item()
                if isinstance(item, dict):
                    # Try multiple field names
                    for field in ["images", "image", "image_path"]:
                        if field in item:
                            img_val = item[field]
                            if isinstance(img_val, (list, tuple)) and len(img_val) > 0:
                                img_path = img_val[0] if isinstance(img_val[0], str) else str(img_val[0])
                                if img_path:
                                    reward_input["image_path"] = img_path
                                    break
                            elif isinstance(img_val, str) and img_val:
                                reward_input["image_path"] = img_val
                                break
            
            # If Code' exists, add to reward_input (for Cycle Consistency RL)
            # Only accept valid strings, avoid garbage values (e.g., [], None, numpy array, etc.)
            if code_prime_list is not None and i < len(code_prime_list):
                code_prime_item = code_prime_list[i]
                if hasattr(code_prime_item, 'item'):
                    code_prime_item = code_prime_item.item()
                
                # Strict check: only accept non-empty strings
                if isinstance(code_prime_item, str) and code_prime_item.strip():
                    reward_input["code_prime"] = code_prime_item
                # Other cases (None, [], numpy array, etc.) are not added to avoid garbage values
            
            reward_inputs.append(reward_input)

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
