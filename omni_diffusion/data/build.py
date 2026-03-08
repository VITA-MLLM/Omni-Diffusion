import logging
import os
from dataclasses import dataclass

from datasets import concatenate_datasets, load_dataset

from .data_collator import DataCollatorForSupervisedDataset, collate_fn_deepspeed
from .dataset_qwen2 import Qwen2Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_supervised_dataset_deepspeed(
    model_config,
    model_args,
    data_args,
    training_args,
    tokenizer,
    create_position_ids=True,
    create_loss_mask=False,
    shift_token=False,
):
    """
    Builds the supervised dataset and data collator specifically for DeepSpeed training.
    
    Args:
        model_config: Configuration object for the model.
        model_args: Arguments related to model architecture and parameters.
        data_args: Arguments related to data processing and loading.
        training_args: Arguments related to the training loop (e.g., output dir, seed).
        tokenizer: The tokenizer instance to process text.
        create_position_ids (bool): Whether to generate position IDs.
        create_loss_mask (bool): Whether to generate a mask for loss calculation.
        shift_token (bool): Whether to shift tokens for causal language modeling.

    Returns:
        dict: A dictionary containing 'train', 'validation' datasets and the 'data_collator'.
    """
    
    logging.info("building dataset...")

    cfg_path = data_args.dataset_name
    max_padding_length = model_args.model_max_length
    output_dir = training_args.output_dir

    create_attention_mask = data_args.create_attention_mask
    create_attention_mask_2d = data_args.create_attention_mask_2d

    image_size = model_args.image_size
    image_token_length = model_args.image_token_length

    max_num_frame = model_args.max_num_frame
    max_fps = model_args.max_fps

    reset_position_ids = data_args.reset_position_ids
    reset_attention_mask = data_args.reset_attention_mask
    variable_length = data_args.variable_length

    min_patch_grid = model_args.min_patch_grid
    max_patch_grid = model_args.max_patch_grid
    process_type = model_args.vision_process_type
    normalize_type = model_args.vision_normalize_type

    audio_tokenizer_path = model_args.audio_tokenizer_path
    audio_tokenizer_type = model_args.audio_tokenizer_type
    image_tokenizer_path = model_args.image_tokenizer_path

    seed = training_args.seed
    cross_dataset_joint = data_args.cross_dataset_joint
    dataset_joint = data_args.dataset_joint

    TrainDataset = Qwen2Dataset

    train_dataset = TrainDataset(
        cfg_path,
        tokenizer,
        image_size=image_size,
        image_token_length=image_token_length,
        max_padding_length=max_padding_length,
        variable_length=variable_length,
        output_dir=output_dir,
        training_args=None,
        shift_token=shift_token,
        create_position_ids=create_position_ids,
        create_attention_mask=create_attention_mask,
        create_attention_mask_2d=create_attention_mask_2d,
        create_loss_mask=create_loss_mask,
        max_num_frame=max_num_frame,
        max_fps=max_fps,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        min_patch_grid=min_patch_grid,
        max_patch_grid=max_patch_grid,
        process_type=process_type,
        normalize_type=normalize_type,
        seed=seed,
        cross_dataset_joint=cross_dataset_joint,
        dataset_joint=dataset_joint,
        audio_tokenizer_type=audio_tokenizer_type,
        audio_tokenizer_path=audio_tokenizer_path,
        image_tokenizer_path=image_tokenizer_path,
        use_megatron=False,
    )
    eval_dataset = None

    data_collator = collate_fn_deepspeed

    return dict(train=train_dataset, validation=eval_dataset, data_collator=data_collator)

