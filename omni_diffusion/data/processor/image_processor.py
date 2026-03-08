import math
import os

import numpy as np
import torch
from PIL import Image

import decord
import natsort
from omni_diffusion.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
)
from torchvision import transforms
from omni_diffusion.tokenizer_magvitv2 import MagVITV2Tokenizer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def image_transform(image, resolution=256):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image


class ImageProcessor:
    def __init__(
        self,
        model_path,
        process_type,
        image_size=256,
        normalize_type="imagenet",
        min_patch_grid=1,
        max_patch_grid=6,
    ):
        self.process_type = process_type
        self.image_size = image_size

        if normalize_type == "imagenet":
            MEAN, STD = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        elif normalize_type == "clip":
            MEAN, STD = OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
        elif normalize_type == "siglip":
            MEAN, STD = IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
        else:
            raise NotImplementedError(normalize_type)
        self.mean = MEAN
        self.std = STD

        self.patch_size = image_size
        self.min_patch_grid = min_patch_grid
        self.max_patch_grid = max_patch_grid

        if self.process_type == "anyres":
            self.grid_pinpoints = [
                (i, j)
                for i in range(min_patch_grid, max_patch_grid + 1)
                for j in range(min_patch_grid, max_patch_grid + 1)
            ]
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.grid_pinpoints
            ]
            print(f"grid_pinpoints {self.grid_pinpoints}")
            print(f"possible_resolutions {self.possible_resolutions}")

        if self.process_type == "dynamic":
            max_num = self.max_patch_grid
            min_num = self.min_patch_grid
            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            self.target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.target_ratios
            ]
            print(f"target_ratios {self.target_ratios}")
            print(f"possible_resolutions {self.possible_resolutions}")

        # self.image_tokenizer = MAGVITv2()
        self.image_tokenizer = MagVITV2Tokenizer(model_path=model_path)

    def load_model(self):
        if self.image_tokenizer is not None:
            self.image_tokenizer.load_model()
        
    def process_images(self, img_or_path_list, image_resolution):

        if isinstance(img_or_path_list[0], str):
            images = [Image.open(x).convert("RGB") for x in img_or_path_list]
        elif isinstance(img_or_path_list[0], Image.Image):
            images = [x.convert("RGB") for x in img_or_path_list]
        else:
            images = img_or_path_list

        # image_tensor = torch.ones([len(images), 3, self.image_size, self.image_size])
        image_tensor = torch.ones([len(images), 3, image_resolution, image_resolution])

        for i, image in enumerate(images):
            image = image_transform(image, resolution=image_resolution)
            
            image_tensor[i] = image

        return image_tensor

    def process_images_with_subpatch(self, img_or_path, image_resolution):
        return self.process_images([img_or_path], image_resolution)
    
    def get_image_token(self, image):
        return self.image_tokenizer.encode(image)
    

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), 
            (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def add_image_input_contiguous(input_ids, image_paths, tokenizer):

    image_processor = ImageProcessor(
        process_type="dynamic",
        image_size=448,
        normalize_type="imagenet",
        min_patch_grid=1,
        max_patch_grid=12,
    )

    image_token_length = 256
    max_num_frame = 4096
    max_fps = 1

    from ...constants import (
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_TAG_TOKEN,
        IMG_CONTEXT_TOKEN,
        VID_START_TOKEN,
        VID_END_TOKEN,
        VID_TAG_TOKEN,
        VID_CONTEXT_TOKEN,
        PATCH_START_TOKEN,
        PATCH_END_TOKEN,
        PATCH_CONTEXT_TOKEN,
    )

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
    VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids

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

    IMG_TAG_ID = IMG_TAG_ID[0]
    VID_TAG_ID = VID_TAG_ID[0]

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

    img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]

    images = []
    image_indices = []
    new_input_ids = []
    st = 0
    for img_idx, img_pos in enumerate(img_positions):
        image_patches, (
            best_width,
            best_height,
        ) = image_processor.process_images_with_subpatch(image_paths[img_idx])
        images.append(image_patches)
        print(f"add_image_input_contiguous best_width {best_width} best_height {best_height}")

        new_input_ids += input_ids[st:img_pos]

        new_input_ids += [IMG_START_ID]

        image_indice_b = torch.zeros(
            1, image_token_length, dtype=torch.int64
        )  # This will change in collate_fn
        image_indice_s = (
            torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
            .unsqueeze(0)
            .repeat(1, 1)
        )
        image_indice_b_s = torch.stack(
            [image_indice_b, image_indice_s], dim=0
        )  # 2, num_image, image_length
        image_indices.append(image_indice_b_s)

        new_input_ids += [IMG_CONTEXT_ID] * image_token_length

        new_input_ids += [IMG_END_ID]

        if len(image_patches) > 1:
            for i in range(0, best_height, image_processor.patch_size):
                new_input_ids += nl_tokens

                for j in range(0, best_width, image_processor.patch_size):
                    new_input_ids += [PATCH_START_ID]

                    image_indice_b = torch.zeros(
                        1, image_token_length, dtype=torch.int64
                    )  # This will change in collate_fn
                    image_indice_s = (
                        torch.arange(
                            len(new_input_ids), len(new_input_ids) + image_token_length
                        )
                        .unsqueeze(0)
                        .repeat(1, 1)
                    )
                    image_indice_b_s = torch.stack(
                        [image_indice_b, image_indice_s], dim=0
                    )  # 2, num_image, image_length
                    image_indices.append(image_indice_b_s)

                    new_input_ids += [PATCH_CONTEXT_ID] * image_token_length

                    new_input_ids += [PATCH_END_ID]
                    # print(f"get_external_dict i {i} j {j} new_input_ids {len(new_input_ids)}")

        st = img_pos + 1

    new_input_ids += input_ids[st:]

    inputs_ids = new_input_ids

    images = torch.cat(images, dim=0)
    image_indices = torch.cat(image_indices, dim=1)

    image_indices = image_indices.contiguous().to(torch.cuda.current_device())
    if True:
        images = (
            torch.tensor(images, dtype=torch.bfloat16).contiguous().to(torch.cuda.current_device())
        )

    else:
        images = (
            torch.tensor(images, dtype=torch.float16).contiguous().to(torch.cuda.current_device())
        )

    return inputs_ids, images, image_indices
