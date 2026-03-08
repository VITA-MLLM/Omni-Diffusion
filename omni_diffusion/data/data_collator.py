import itertools
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import default_collate
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Data collator for supervised fine-tuning.
    Handles padding of input IDs and labels, creation of attention masks,
    and concatenation of multimodal inputs (images).
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_TOKEN_ID,
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "images" in instances[0]:
            images = [instance["images"] for instance in instances]
            batch["images"] = torch.cat(images, dim=0)

        if "doclm_images" in instances[0]:
            images = [instance["doclm_images"] for instance in instances]
            batch["doclm_images"] = torch.cat(images, dim=0)

        if "image_paths" in instances[0]:
            image_paths = [instance["image_paths"] for instance in instances]
            batch["image_paths"] = image_paths

        if "pixel_values" in instances[0]:
            pixel_values = torch.cat([instance["pixel_values"] for instance in instances])
            batch["pixel_values"] = pixel_values

        if "image_flags" in instances[0]:
            image_flags = torch.cat([instance["image_flags"] for instance in instances])
            batch["image_flags"] = image_flags

        return batch

def collate_fn_deepspeed(batch):
    """
    Custom collation function designed for DeepSpeed training.
    Handles complex multimodal data (images/audio) and potentially packed sequences (cu_seq_lens).
    """
    tmp_batch = [{} for _ in range(len(batch))]
    if "cu_seq_lens" in batch[0]:
        cu_seq_lens = [x["cu_seq_lens"] for x in batch]
        max_seq_len = [x["max_seq_len"] for x in batch]
    else:
        cu_seq_lens = None
        max_seq_len = None

    if "images" in batch[0].keys():
        for new_x, x in zip(tmp_batch, batch):
            new_x["images"] = x.pop("images")
            new_x["image_indices"] = x.pop("image_indices")

    if "audios" in batch[0].keys():
        for new_x, x in zip(tmp_batch, batch):
            new_x["audios"] = x.pop("audios")
            new_x["audio_indices"] = x.pop("audio_indices")

    new_batch = default_collate(batch)

    if "images" in tmp_batch[0].keys():
        
        new_batch["images"] = torch.cat([x["images"] for x in tmp_batch], dim=0)

        for sample_idx, sample in enumerate(tmp_batch):
            sample["image_indices"][0, :, :] = sample_idx

        new_batch["image_indices"] = torch.cat([x["image_indices"] for x in tmp_batch], dim=1)

    if "audios" in tmp_batch[0].keys():
        
        new_batch["audios"] = list(itertools.chain.from_iterable([x["audios"] for x in tmp_batch]))
        # print(f"{[x.size() for x in sample['audios']]}")

        for sample_idx, sample in enumerate(tmp_batch):
            for j in range(len(sample["audio_indices"])):
                sample["audio_indices"][j][0, :, :] = sample_idx

        new_batch["audio_indices"] = list(
            itertools.chain.from_iterable([x["audio_indices"] for x in tmp_batch])
        )
        # print(f"{[x.size() for x in sample['audio_indices']]}")

    
    return new_batch
