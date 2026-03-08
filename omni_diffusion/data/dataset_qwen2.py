import json
import logging
import math
import os
import pdb
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Sequence
from typing import List, Optional, Tuple, Union
import copy
import numpy as np
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import soundfile as sf

from .dataset_base import BaseDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def forward_process(
    bsz: int,
    seq_len: int,
    device: torch.device,
    labels: torch.Tensor,                
    eps: float = 1e-3,
    special_token_id: int = 151643,     
    special_mask_ratio: float = 0.1      
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a mask for the input sequences, applying different masking probabilities 
    for normal tokens and special tokens within the valid (non-padded) regions.
    """

    # Initialize the boolean mask matrix and the overall mask ratio for each sample
    b, l = bsz, seq_len
    masked_indices = torch.zeros((b, l), device=device, dtype=torch.bool)
    p_mask = torch.rand(b, device=device)
    p_mask = (1 - eps) * p_mask + eps
    p_mask = p_mask.unsqueeze(1)  # [b,1]

    # Find the first and last valid positions (where label is not -100) for each sequence
    first_idxs = []
    last_idxs  = []
    for i in range(b):
        # -100 is the ignore_index
        nonneg = (labels[i] != -100).nonzero(as_tuple=True)[0]
        if nonneg.numel() == 0:
            first_idxs.append(None)
            last_idxs.append(None)
        else:
            first_idxs.append(int(nonneg[0]))
            last_idxs.append(int(nonneg[-1]))

    # Generate masks specifically for the valid interval of each sequence
    for i in range(b):
        start = first_idxs[i]
        end   = last_idxs[i]

        # Skip if the sequence has no valid tokens or invalid boundaries
        if start is None or end is None or end < start:
            continue

        # Generate base thresholds for each position in the valid interval
        valid_len = end - start + 1
        t = torch.rand(valid_len, device=device)
        mask_threshold = (1 - eps) * t + eps  # [valid_len]

        # Generate random values to make masking decisions
        rand_vals = torch.rand(valid_len, device=device)
        # Normal token masking decision
        normal_mask = rand_vals <= mask_threshold
        # Special token masking threshold is lower
        special_thresh = mask_threshold * special_mask_ratio
        special_mask   = rand_vals <= special_thresh

        labels_slice = labels[i, start : end + 1]
        # Final mask: special tokens use special_mask, others use normal_mask
        final_mask = torch.where(
            labels_slice == special_token_id,
            special_mask,
            normal_mask
        )

        masked_indices[i, start : end + 1] = final_mask

    total_masked   = int(masked_indices.sum().item())
    special_masked = int((masked_indices & (labels == special_token_id)).sum().item())

    return masked_indices, p_mask

def update_labels(input_ids, labels, eos_id, max_n=20):
    """
    Finds the first occurrence of the EOS token in each sequence and updates 
    up to `max_n` subsequent labels to the EOS token ID.
    """
    batch_size, seq_len = input_ids.shape
    first_occurrence_indices = []

    # Record the first occurrence position of eos_id in each batch sample
    for idx in range(batch_size):
        eos_positions = (input_ids[idx] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_occurrence_indices.append(eos_positions[0].item())
        else:
            first_occurrence_indices.append(-1)  

    # Select a random number of sequential positions (up to max_n) starting from first_idx to update
    for i in range(batch_size):
        first_idx = first_occurrence_indices[i]
        if first_idx == -1:
            continue 
        max_possible = seq_len - first_idx

        if max_possible <= 0:
            continue
        num_to_select = random.randint(1, min(max_n, max_possible))

        selected_indices = torch.arange(first_idx, first_idx + num_to_select)

        labels[i, selected_indices] = eos_id

    return labels


import torch
import random

def update_labels_and_inputs(input_ids, labels, eos_id, max_n=20):
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    labels = torch.tensor(labels).unsqueeze(0)
    batch_size, seq_len = input_ids.shape
    input_ids = input_ids.clone()
    labels = labels.clone()
    new_input_ids = []
    new_labels = []

    for idx in range(batch_size):
        eos_positions = (input_ids[idx] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            first_idx = eos_positions[0].item()
            cur_input_ids = input_ids[idx]
            cur_labels = labels[idx]
        else:
            random_max_n = random.randint(1, max_n)
            eos_ids = torch.full((random_max_n,), eos_id, device=input_ids.device, dtype=input_ids.dtype)
            cur_input_ids = torch.cat([input_ids[idx], eos_ids])
            pad_labels = torch.full((random_max_n,), eos_id, device=labels.device, dtype=labels.dtype)
            cur_labels = torch.cat([labels[idx], pad_labels])
            # first_idx = len(cur_input_ids) - random_max_n


        new_input_ids.append(cur_input_ids)
        new_labels.append(cur_labels)

    max_len = max(len(x) for x in new_input_ids)
    padded_input_ids = torch.stack([
        torch.cat([x, torch.full((max_len - len(x),), eos_id, device=x.device, dtype=x.dtype)])
        for x in new_input_ids
    ])
    padded_labels = torch.stack([
        torch.cat([x, torch.full((max_len - len(x),), eos_id, device=x.device, dtype=x.dtype)])
        for x in new_labels
    ])

    return padded_input_ids, padded_labels



def pad_or_truncate_to_512(
        input_ids, 
        labels,
        eos_id,
        target_len: int = 512,
):

    input_ids = torch.as_tensor(input_ids).unsqueeze(0)   # (B, L)
    labels    = torch.as_tensor(labels   ).unsqueeze(0)

    batch_size = input_ids.size(0)
    new_input_ids, new_labels = [], []

    for i in range(batch_size):
        cur_input = input_ids[i]
        cur_label = labels[i]

        cur_input = cur_input[:target_len]
        cur_label = cur_label[:target_len]

        pad_len = target_len - cur_input.size(0)
        if pad_len > 0:
            eos_pad   = torch.full((pad_len,), eos_id, device=cur_input.device, dtype=cur_input.dtype)
            label_pad = torch.full((pad_len,), eos_id, device=cur_label.device, dtype=cur_label.dtype)
            cur_input = torch.cat([cur_input, eos_pad])
            cur_label = torch.cat([cur_label, label_pad])

        new_input_ids.append(cur_input)
        new_labels.append(cur_label)

    return torch.stack(new_input_ids), torch.stack(new_labels)



class Qwen2Dataset(BaseDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.default_system_message = "You are a helpful AI assistant."
        self.default_system_message = None

        self.ret = defaultdict(dict)
        self.is_cat = True

        if self.cross_dataset_joint:
            for i in range(2):
                self.maybe_init_ret(f"default_{i}")

    def maybe_init_ret(self, source, force=False):
        """Initializes the return dictionary buffer for a specific source if it doesn't exist."""
        if source not in self.ret or force:
            self.ret[source] = {}

            self.ret[source]["tokens"] = []
            self.ret[source]["labels"] = []
            self.ret[source]["cu_seq_lens"] = [0]
            self.ret[source]["max_seq_len"] = self.max_padding_length

            if self.create_position_ids:
                self.ret[source]["position_ids"] = []

            if self.create_attention_mask:
                self.ret[source]["attention_mask"] = []

            if self.create_attention_mask_2d:
                self.ret[source]["attention_mask_2d"] = torch.tril(
                    torch.ones(
                        (1, self.max_padding_length, self.max_padding_length), dtype=torch.bool
                    )
                )
        return len(self.ret[source]["tokens"]) == 0

    def get_max_min_ret_length(self):
        """Finds the buffer with the maximum and minimum current lengths."""
        max_ret_lengh = 0
        min_ret_lengh = self.max_padding_length + 1

        max_ret_key = None
        min_ret_key = None

        for k, v in self.ret.items():
            cur_length = len(v["tokens"])

            if cur_length > max_ret_lengh:
                max_ret_lengh = cur_length
                max_ret_key = k

            if cur_length < min_ret_lengh:
                min_ret_lengh = cur_length
                min_ret_key = k

        return max_ret_lengh, max_ret_key, min_ret_lengh, min_ret_key

    def add_ret(self, ret, source):
        """
        Appends a processed sample (tokens, images, audios) to the buffer (`self.ret`).
        Updates indices for images/audios to account for the offset in the concatenated sequence.
        """
        cur_length = len(ret["input_ids"])
        cur_image_length = len(ret["images"])
        cur_audio_length = len(ret["audios"])

        all_length = len(self.ret[source]["tokens"])

        if "images" in self.ret[source]:
            all_image_length = len(self.ret[source]["images"])
        else:
            all_image_length = 0

        if cur_image_length > 0:
            if all_image_length > 0:
                self.ret[source]["images"] = torch.cat(
                    [self.ret[source]["images"], ret["images"]], dim=0
                )
                ret["image_indices"][1, :, :] += all_length
                self.ret[source]["image_indices"] = torch.cat(
                    [self.ret[source]["image_indices"], ret["image_indices"]], dim=1
                )
            else:
                self.ret[source]["images"] = ret["images"]
                self.ret[source]["image_indices"] = ret["image_indices"]

        if "audios" in self.ret[source]:
            all_audio_length = len(self.ret[source]["audios"])
        else:
            all_audio_length = 0

        if cur_audio_length > 0:
            if all_audio_length > 0:
                self.ret[source]["audios"].extend(ret["audios"])
                for audio_indice in ret["audio_indices"]:
                    audio_indice[1, :, :] += all_length
                self.ret[source]["audio_indices"].extend(ret["audio_indices"])
            else:
                self.ret[source]["audios"] = ret["audios"]
                self.ret[source]["audio_indices"] = ret["audio_indices"]

            # print(self.ret[source]["audios"])

        if self.create_attention_mask:
            self.ret[source]["attention_mask"] += ret["attention_mask"]

        if self.create_attention_mask_2d:
            self.ret[source]["attention_mask_2d"][:, all_length:, :all_length] = 0

        if self.create_position_ids:
            self.ret[source]["position_ids"] += list(range(cur_length))

        self.ret[source]["tokens"] += ret["input_ids"]
        self.ret[source]["labels"] += ret["labels"]
        self.ret[source]["cu_seq_lens"] += [all_length + cur_length]

    def process_ret(self, to_ret):
        """
        Finalizes the buffer for return. Handles padding, truncation, tensor conversion,
        and attention mask generation.
        """
        if "tokens" in to_ret and len(to_ret["tokens"]) > 0:
            pass
        else:
            return to_ret

        if self.create_position_ids:
            if self.reset_position_ids:
                pass
            else:
                to_ret["position_ids"] = list(range(len(to_ret["tokens"])))

        if self.create_attention_mask_2d:
            if self.reset_attention_mask:
                pass
            else:
                to_ret["attention_mask_2d"] = torch.tril(
                    torch.ones(
                        (1, self.max_padding_length, self.max_padding_length), dtype=torch.bool
                    )
                )

        if self.shift_token:
            to_ret["tokens"] = to_ret["tokens"][:-1]
            to_ret["labels"] = to_ret["labels"][1:]
            to_ret["cu_seq_lens"][-1] -= 1
            if self.create_position_ids:
                to_ret["position_ids"] = to_ret["position_ids"][:-1]
            if self.create_attention_mask:
                to_ret["attention_mask"] = to_ret["attention_mask"][:-1]

            if self.create_attention_mask_2d:
                to_ret["attention_mask_2d"][:, :, -1] = 0
                to_ret["attention_mask_2d"][:, -1, :] = 0

        assert len(to_ret["tokens"]) == len(
            to_ret["labels"]
        ), f"{len(to_ret['tokens'])} {len(to_ret['labels'])}"

        if not self.variable_length and self.max_padding_length > len(to_ret["tokens"]):
            to_ret["tokens"] += [self.tokenizer.pad_token_id] * (
                self.max_padding_length - len(to_ret["tokens"])
            )
            to_ret["labels"] += [IGNORE_TOKEN_ID] * (
                self.max_padding_length - len(to_ret["labels"])
            )
            to_ret["cu_seq_lens"][-1] = self.max_padding_length
            if self.create_position_ids:
                # to_ret["position_ids"] += to_ret["position_ids"][-1:] * (
                #     self.max_padding_length - len(to_ret["position_ids"])
                # )
                to_ret["position_ids"] += list(
                    range(to_ret["position_ids"][-1] + 1, self.max_padding_length)
                )
            if self.create_attention_mask:
                to_ret["attention_mask"] += [0] * (
                    self.max_padding_length - len(to_ret["attention_mask"])
                )

        to_ret["tokens"] = to_ret["tokens"][: self.max_padding_length]
        to_ret["labels"] = to_ret["labels"][: self.max_padding_length]
        to_ret["cu_seq_lens"][-1] = self.max_padding_length

        if self.create_position_ids:
            to_ret["position_ids"] = to_ret["position_ids"][: self.max_padding_length]
        if self.create_attention_mask:
            to_ret["attention_mask"] = to_ret["attention_mask"][: self.max_padding_length]

        to_ret["tokens"] = torch.tensor(to_ret["tokens"], dtype=torch.int64)
        to_ret["labels"] = torch.tensor(to_ret["labels"], dtype=torch.int64)
        to_ret["cu_seq_lens"] = torch.tensor(to_ret["cu_seq_lens"], dtype=torch.int64)
        if self.create_position_ids:
            to_ret["position_ids"] = torch.tensor(to_ret["position_ids"], dtype=torch.int64)
        if self.create_attention_mask:
            to_ret["attention_mask"] = torch.tensor(to_ret["attention_mask"], dtype=torch.int64)

        if self.create_attention_mask_2d:
            attention_mask_2d = to_ret.pop("attention_mask_2d")
            attention_mask_2d = attention_mask_2d.masked_fill(
                (to_ret["attention_mask"] < 0.5).view(1, 1, self.max_padding_length), value=0
            )
            attention_mask_2d = attention_mask_2d < 0.5

            to_ret["attention_mask"] = attention_mask_2d

        if self.create_loss_mask:
            loss_mask = torch.where(to_ret["labels"] == IGNORE_TOKEN_ID, 0, 1)
            to_ret["loss_mask"] = loss_mask.to(torch.float32)

        if not self.reset_position_ids and not self.reset_attention_mask:
            to_ret.pop("cu_seq_lens")
        else:
            max_seq_len = max(to_ret["cu_seq_lens"][1:] - to_ret["cu_seq_lens"][:-1])
            to_ret["max_seq_len"] = max_seq_len
        to_ret["input_ids"] = to_ret["tokens"]

        return to_ret

    def is_skip(self):

        processed_samples = sum(self.processed_samples.values())
        if processed_samples < self.skip_samples:
            if processed_samples % 1e3 == 0:
                print(
                    f"processed_samples {processed_samples} skip_samples {self.skip_samples}"
                )
            return True

    def show_statistic(self):
        log_interval = 2000
        if self.max_padding_length >= 2**17:
            log_interval = 1000
        if self.max_padding_length >= 2**20:
            log_interval = 200 

        processed_samples = sum(self.processed_samples.values())
        unjoint_samples = sum(self.unjoint_samples.values())
        joint_samples = sum(self.joint_samples.values())
        if unjoint_samples % log_interval == 1:
            pass
        else:
            return

        with open(os.path.join(self.output_dir, "data_statistics.log"), "a") as f:
            print("-" * 100, file=f)
            print(
                f"processed_samples {processed_samples}" +
                f" unjoint_samples {unjoint_samples}" +
                f" joint_samples {joint_samples}" +
                f" {[len(v['tokens']) for _, v in self.ret.items()]}", file=f,
            )

            print("source processed_samples  unjoint_samples    joint_samples data_path", file=f)
            for source, data_path in self.source2jsonpath.items():
                print(f"{source: >6}  {self.processed_samples[source]: >16} \
                      {self.unjoint_samples[source]: >16} {self.joint_samples[source]: >16} {data_path}", file=f)

    def __getitem__(self, index):
        """
        Main data loading function.
        1. Fetches raw data.
        2. Preprocesses it (tokenization, multimodal handling).
        3. Packs multiple samples into a single buffer (`self.ret`) until `max_padding_length` is reached.
        4. Returns the packed batch when full.
        """
        self.processor["audio"].load_model()
        self.processor["image"].load_model()

        while True:
            try:
                sample = self.raw_data[index]
                sample = copy.deepcopy(sample)
                sample = self.update_data_path(sample)
                source = sample["source"]

                self.processed_samples[source] += 1
                if self.is_skip():
                    return {}

                if self.cross_dataset_joint:
                    is_empty = False
                    (
                        max_ret_lengh,
                        max_ret_key,
                        min_ret_lengh,
                        min_ret_key,
                    ) = self.get_max_min_ret_length()
                else:
                    is_empty = self.maybe_init_ret(source)

                    max_ret_lengh = min_ret_lengh = len(self.ret[source]["tokens"])
                    max_ret_key = min_ret_key = source

                is_begin = is_empty or self.reset_position_ids or self.reset_attention_mask

                #logger.info("preprocess begin " + str(self.processor["audio"].audio_tokenizer.device))

                ret = preprocess(
                    sample,
                    self.tokenizer,
                    self.image_token_length,
                    default_system_message=self.default_system_message,
                    processor=self.processor,
                    is_begin=is_begin,
                    max_num_frame=self.max_num_frame,
                    max_fps=self.max_fps,
                )

                #logger.info("preprocess end " + str(self.processor["audio"].audio_tokenizer.device))

                if ret is None:
                    #logger.info("ret is None " + str(self.processor["audio"].audio_tokenizer.device))
                    return {}

                cur_length = len(ret["input_ids"])

                if cur_length > self.max_padding_length:
                    return {}

                self.unjoint_samples[source] += 1

                if not self.dataset_joint:
                    import pdb; pdb.set_trace()
                    to_ret = self.ret.pop(max_ret_key)

                    self.maybe_init_ret(max_ret_key, force=True)
                    self.add_ret(ret, max_ret_key)

                elif min_ret_lengh + cur_length > self.max_padding_length:
                    #logger.info("data too long " + str(self.processor["audio"].audio_tokenizer.device))
                    to_ret = self.ret.pop(max_ret_key)
                    self.joint_samples[source] += 1

                    self.maybe_init_ret(max_ret_key, force=True)
                    self.add_ret(ret, max_ret_key)

                else:
                    to_ret = {}
                    self.add_ret(ret, min_ret_key)

                to_ret = self.process_ret(to_ret)

                self.show_statistic()
                #logger.info("output ret " + str(len(ret)) + " " + str(self.processor["audio"].audio_tokenizer.device))
                return to_ret

            except Exception as error:
                try:
                    with open(os.path.join(self.output_dir, "data_error.log"), "a") as f:
                        #print("-" * 100)
                        #print(traceback.format_exc())
                        #print(self.raw_data[index])
                        print("-" * 100, file=f)
                        print(traceback.format_exc(), file=f)
                        print(self.raw_data[index], file=f)
                except Exception as error:
                    print(error)
                return {}


def preprocess(
    sample,
    tokenizer: transformers.PreTrainedTokenizer,
    default_system_message: str = "You are a helpful assistant.",
    processor=None,
    is_begin: bool = True,
) -> Dict:

    from ..constants import (
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        VID_START_TOKEN,
        VID_END_TOKEN,
        VID_CONTEXT_TOKEN,
        PATCH_START_TOKEN,
        PATCH_END_TOKEN,
        PATCH_CONTEXT_TOKEN,
        AUD_START_TOKEN,
        AUD_END_TOKEN,
        IMG_TAG_TOKEN,
        VID_TAG_TOKEN,
        AUD_TAG_TOKEN,
        AUD_CONTEXT_TOKEN,
    )

    human_roles = ["user", "human"]
    gpt_roles = ["assistant", "gpt"]
    system_roles = ["system"]

    # Ensure special tokens map to exactly one ID
    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
    AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids

    IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
    VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    assert len(VID_CONTEXT_ID) == 1
    assert len(VID_START_ID) == 1
    assert len(VID_END_ID) == 1

    assert len(PATCH_CONTEXT_ID) == 1
    assert len(PATCH_START_ID) == 1
    assert len(PATCH_END_ID) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    VID_CONTEXT_ID = VID_CONTEXT_ID[0]
    VID_START_ID = VID_START_ID[0]
    VID_END_ID = VID_END_ID[0]

    PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
    PATCH_START_ID = PATCH_START_ID[0]
    PATCH_END_ID = PATCH_END_ID[0]

    AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
    AUD_START_ID = AUD_START_ID[0]
    AUD_END_ID = AUD_END_ID[0]

    IMG_TAG_ID = IMG_TAG_ID[0]
    VID_TAG_ID = VID_TAG_ID[0]
    AUD_TAG_ID = AUD_TAG_ID[0]

    BOS_ID = tokenizer.bos_token_id
    EOS_ID = tokenizer.eos_token_id

    # ChatML format special tokens
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
    IM_START_IDS = tokenizer(IM_START, add_special_tokens=False).input_ids
    IM_END_IDS = tokenizer(IM_END, add_special_tokens=False).input_ids
    USER_IDS = tokenizer(USER, add_special_tokens=False).input_ids
    ASSISTANT_IDS = tokenizer(ASSISTANT, add_special_tokens=False).input_ids
    SYSTEM_IDS = tokenizer(SYSTEM, add_special_tokens=False).input_ids

    input_ids, targets = [], []
    images = []
    image_indices = []
    audios = []
    audio_indices = []

    messages = []
    if "conversations" in sample:
        messages = sample["conversations"]
    if len(messages) == 0 and "messages" in sample:
        messages = sample["messages"]

    # ----------------------------------------------------------------
    # 1. System Prompt Processing
    # ----------------------------------------------------------------
    has_system = False
    if is_begin:
        if messages[0]["role"] == "system":
            has_system = True
        else:
            has_system = False

        if (
            not has_system
            and default_system_message is not None
            and len(default_system_message) > 0
        ):
            messages = [{"role": "system", "content": default_system_message}] + messages
            has_system = True

    # ----------------------------------------------------------------
    # 2. Audio Processing
    # ----------------------------------------------------------------
    if has_audio(sample) and processor["audio"].is_discrete:
        unused_audio_idxs = list(range(len(sample["audios"])))
        
        audio_tokens_list = [
            processor["audio"].process_audios(x, is_discrete=True) for x in sample["audios"]
        ]
        audio_tokens_list = ["".join(f"<|audio_{i}|>" for i in x) for x in audio_tokens_list]

        audio_idx = 0
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            role = sentence["role"]
            # whether apply discrete tokenize to this role
            if processor["audio"].apply_to_role(role, is_discrete=True):
                while AUD_TAG_TOKEN in content:
                    content = content.replace(
                        AUD_TAG_TOKEN,
                        f"{AUD_START_TOKEN}{audio_tokens_list[audio_idx]}{AUD_END_TOKEN}",
                        1,
                    )
                    # <|begin_of_audio|> <|audio_0|> <|audio_1|> ... <|audio_n|> <|end_of_audio|>
                    unused_audio_idxs.remove(audio_idx)
                    audio_idx += 1
            else:
                audio_idx += content.count(AUD_TAG_TOKEN)

            sentence["content"] = content

    # ----------------------------------------------------------------
    # 3. Image Processing
    # ----------------------------------------------------------------
    if has_image(sample):
        # for visual question answering & captioning
        image_tokens_512_list = [
            processor["image"].process_images_with_subpatch(x, 512) for x in sample["images"]
        ]
        image_tokens_512_list = [
            processor["image"].get_image_token(x) for x in image_tokens_512_list
        ]
        image_tokens_512_list = [x[0].tolist() for x in image_tokens_512_list]
        image_tokens_512_list = ["".join(f"<|image_{i}|>" for i in x) for x in image_tokens_512_list]

        # for image generation
        image_tokens_256_list = [
            processor["image"].process_images_with_subpatch(x, 256) for x in sample["images"]
        ]
        image_tokens_256_list = [
            processor["image"].get_image_token(x) for x in image_tokens_256_list
        ]
        image_tokens_256_list = [x[0].tolist() for x in image_tokens_256_list]
        image_tokens_256_list = ["".join(f"<|image_{i}|>" for i in x) for x in image_tokens_256_list]

        image_idx = 0
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            role = sentence["role"]
            # for image, always apply discrete tokenize to this role
            # if processor["image"].apply_to_role(role) or True:
            if role == "user":
                image_resolution = 512
            else:
                image_resolution = 256

            while IMG_TAG_TOKEN in content:
                if image_resolution == 256:
                    content = content.replace(
                        IMG_TAG_TOKEN,
                        f"{IMG_START_TOKEN}{image_tokens_256_list[image_idx]}{IMG_END_TOKEN}",
                        1,
                    )
                    # <|begin_of_image|> <|image_0|> <|image_1|> ... <|image_n|> <|end_of_image|>
                else:
                    content = content.replace(
                        IMG_TAG_TOKEN,
                        f"{IMG_START_TOKEN}{image_tokens_512_list[image_idx]}{IMG_END_TOKEN}",
                        1,
                    )
                    # <|begin_of_image|> <|image_0|> <|image_1|> ... <|image_n|> <|end_of_image|>
                image_idx += 1
            else:
                image_idx += content.count(IMG_TAG_TOKEN)

            sentence["content"] = content

    # ----------------------------------------------------------------
    # 4. Text Processing
    # ----------------------------------------------------------------
    for j, sentence in enumerate(messages):
        role = sentence["role"]
        content = sentence["content"]

        if role in human_roles:
            _input_id = (
                IM_START_IDS
                + USER_IDS
                + nl_tokens
                + tokenizer(content, add_special_tokens=False).input_ids
                + IM_END_IDS
                + nl_tokens
            )
            _target = [IGNORE_TOKEN_ID] * len(_input_id)

        elif role in gpt_roles:
            content_input_id = tokenizer(content, add_special_tokens=False).input_ids

            _input_id = (
                IM_START_IDS + ASSISTANT_IDS + nl_tokens + content_input_id + IM_END_IDS + nl_tokens
            )
            _target = (
                [IGNORE_TOKEN_ID] * len(IM_START_IDS)
                + [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS)
                + [IGNORE_TOKEN_ID] * len(nl_tokens)
                + content_input_id
                + IM_END_IDS
                + nl_tokens
            )

            dream_pad_token = tokenizer.encode("<|endoftext|>")
            _input_id_dream = (
                IM_START_IDS + ASSISTANT_IDS + nl_tokens + content_input_id + IM_END_IDS + dream_pad_token
            )
            _target_dream = (
                [IGNORE_TOKEN_ID] * len(IM_START_IDS)
                + [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS)
                + [IGNORE_TOKEN_ID] * len(nl_tokens)
                + content_input_id
                + IM_END_IDS
                + dream_pad_token
            )

        elif role in system_roles:
            _input_id = (
                IM_START_IDS
                + SYSTEM_IDS
                + nl_tokens
                + tokenizer(content, add_special_tokens=False).input_ids
                + IM_END_IDS
                + nl_tokens
            )
            _target = [IGNORE_TOKEN_ID] * len(_input_id)

        else:
            raise NotImplementedError

        input_ids += _input_id
        targets += _target


    # ----------------------------------------------------------------
    # 5. Contiguous Audio Processing
    # ----------------------------------------------------------------
    if has_audio(sample) and processor["audio"].is_contiguous:
        aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
        assert len(aud_positions) == len(unused_audio_idxs), sample

        new_input_ids = []
        new_targets = []
        st = 0

        for aud_idx, aud_pos in enumerate(aud_positions):
            aud_idx = unused_audio_idxs[aud_idx]
            audio = processor["audio"].process_audios(sample["audios"][aud_idx], is_contiguous=True)
            audios.append(audio)
            audio_token_length = audio.size(0) + 4
            # audio_token_length = audio.size(0)

            new_input_ids += input_ids[st:aud_pos]
            new_targets += targets[st:aud_pos]

            new_input_ids += [AUD_START_ID]
            new_targets += [IGNORE_TOKEN_ID]

            audio_indice_b = torch.zeros(
                1, audio_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            audio_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            audio_indice_b_s = torch.stack(
                [audio_indice_b, audio_indice_s], dim=0
            )  # 2, num_image, image_length
            audio_indices.append(audio_indice_b_s)

            new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
            new_targets += [IGNORE_TOKEN_ID] * audio_token_length

            new_input_ids += [AUD_END_ID]
            new_targets += [IGNORE_TOKEN_ID]

            st = aud_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    if len(images) > 0:
        images = torch.cat(images, dim=0)

    if len(image_indices) > 0:
        image_indices = torch.cat(image_indices, dim=1)

    # ----------------------------------------------------------------
    # 6. Final Masking and Label Updating 
    # ----------------------------------------------------------------
    origin_input = input_ids
    labels = targets
    eos_id = 151643               
    mask_id = 151666    
    input_ids, labels = update_labels_and_inputs(input_ids,labels,eos_id,16)

    labels_mask           = ~(labels == -100) 
    bsz, seq_len          = labels_mask.shape
    masked_indices, p_mask = forward_process(
        bsz, seq_len, input_ids.device, labels,special_mask_ratio=0.6, special_token_id=mask_id
    )
    final_masked_indices = masked_indices & labels_mask
    final_masked_indices_inv = (~masked_indices) & labels_mask

    mask_id_tensor = torch.full_like(input_ids, mask_id)
    input_ids = torch.where(final_masked_indices, mask_id_tensor, input_ids)

    new_labels = labels.clone()
    new_labels[final_masked_indices_inv] = -100

    input_ids = input_ids.squeeze(0).cpu().tolist()
    new_labels = new_labels.squeeze(0).cpu().tolist()
    attention_mask = [1] * len(input_ids)
    assert len(new_labels) == len(input_ids)
    
    return dict(
        input_ids=input_ids,
        labels=new_labels,
        attention_mask=attention_mask,
        images=images,
        image_indices=image_indices,
        audios=audios,
        audio_indices=audio_indices,
    )


def has_image(sample):
    # image
    if (
        "images" in sample
        and isinstance(sample["images"], list)
        and None not in sample["images"]
        and len(sample["images"])
    ):
        return True
    return False


def has_audio(sample):
    # audio
    if (
        "audios" in sample
        and isinstance(sample["audios"], list)
        and None not in sample["audios"]
        and len(sample["audios"])
    ):
        return True
    return False
