from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None,
        debug: bool = False,
        fail_on_empty: bool = False,
        fail_on_eos_only: bool = False,
        eos_only_fail_before_step: int = 1,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        self.debug = bool(debug)
        self.fail_on_empty = bool(fail_on_empty)
        self.fail_on_eos_only = bool(fail_on_eos_only)
        self.eos_only_fail_before_step = max(0, int(eos_only_fail_before_step))
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        vocab_size = scores.shape[-1]
        total_beams = 0
        empty_allowed_count = 0
        eos_only_count = 0
        allowed_sizes = []
        masked_sizes = []
            
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                total_beams += 1
                if self.count == 0:
                    hash_key = sent[-self.prefix_index:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = tuple(hash_key.tolist())
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                allowed_size = len(prefix_allowed_tokens)
                allowed_sizes.append(allowed_size)
                masked_sizes.append(vocab_size - allowed_size)

                if allowed_size == 0:
                    empty_allowed_count += 1
                    message = (
                        f"[DEBUG][PROC][FAILFAST] allowed token set is empty at step={self.count}, "
                        f"batch={batch_id}, beam={beam_id}, hash_key={hash_key}. "
                        "这通常表示 trie/hash_dict 构建或 prefix_index 分支错误。"
                    )
                    if self.fail_on_empty:
                        raise RuntimeError(message)
                    warnings.warn(message)
                    # 保持兼容：未开启 fail-fast 时，退化为只允许 EOS 结束无效序列。
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue

                if (
                    allowed_size == 1
                    and self.eos_token_id is not None
                    and prefix_allowed_tokens[0] == self.eos_token_id
                ):
                    eos_only_count += 1
                    message = (
                        f"[DEBUG][PROC] allowed token set only contains EOS at step={self.count}, "
                        f"batch={batch_id}, beam={beam_id}, hash_key={hash_key}."
                    )
                    # 仅在“过早只剩 EOS”时 fail-fast；尾部自然收束到 EOS 是正常现象。
                    is_early_eos_only = self.count <= self.eos_only_fail_before_step
                    if self.fail_on_eos_only and is_early_eos_only:
                        raise RuntimeError(
                            message
                            + " 这是过早 EOS-only，疑似约束过严或 prefix 路径错误。"
                        )
                    if self.debug:
                        print(message + f" early={is_early_eos_only}")
                
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        if self.debug and total_beams > 0:
            min_allowed = min(allowed_sizes) if allowed_sizes else 0
            max_allowed = max(allowed_sizes) if allowed_sizes else 0
            mean_allowed = float(sum(allowed_sizes) / len(allowed_sizes)) if allowed_sizes else 0.0
            mean_masked = float(sum(masked_sizes) / len(masked_sizes)) if masked_sizes else 0.0
            print(
                "[DEBUG][PROC] "
                f"step={self.count}, beams={total_beams}, "
                f"allowed(min/mean/max)={min_allowed}/{mean_allowed:.2f}/{max_allowed}, "
                f"masked_mean={mean_masked:.2f}, "
                f"empty={empty_allowed_count}, eos_only={eos_only_count}"
            )

        self.count += 1

        scores = scores + mask
        return scores
