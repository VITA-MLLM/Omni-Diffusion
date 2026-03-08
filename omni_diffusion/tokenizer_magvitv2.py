import glob
import io
import logging
import math
import os
import tarfile
import uuid

import safetensors
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast

import torchaudio

from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from flow_inference import AudioDecoder
from omni_diffusion.models.magvit.modeling_magvitv2 import MAGVITv2

from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.sense_voice.model import SenseVoiceSmall
            
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MagVITV2Tokenizer:
    def __init__(self, model_path=None, rank=None):
        if rank is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            self.rank = rank % 8
        else:
            self.rank = rank
        logger.info(f"{self.rank=}")
        self.model_path = model_path

    def load_model(self):

        if hasattr(self, "image_tokenizer"):
            return

        if self.rank is not None:
            self.device = f"cuda:{self.rank}"
            torch.cuda.set_device(self.rank)
        else:
            self.device = "cpu"

        self.image_tokenizer = MAGVITv2()
        logger.info(f"{self.device=} Loading MAGVITv2")
        self.image_tokenizer = self.image_tokenizer.from_pretrained(self.model_path).to(self.device)
        self.image_tokenizer.eval()
        self.image_tokenizer.requires_grad_(False) 
        logger.info(f"{self.device=} Loading MAGVITv2 done")

    def encode(self, image):
        image = image.to(self.device)
        image_tokens = self.image_tokenizer.get_code(image)
        return image_tokens

    def decode(self, image_tokens):
        images = self.image_tokenizer.decode_code(image_tokens)
        return images

    def apply_to_role(self, role, **kwargs):
        return True
