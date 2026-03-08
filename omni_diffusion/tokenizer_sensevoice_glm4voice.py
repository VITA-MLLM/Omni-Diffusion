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

from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.sense_voice.model import SenseVoiceSmall
import sys
            
from .constants import (
    AUD_CONTEXT_TOKEN,
    AUD_END_TOKEN,
    AUD_START_TOKEN,
    AUD_TAG_TOKEN,
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    IMG_TAG_TOKEN,
    PATCH_CONTEXT_TOKEN,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    VID_CONTEXT_TOKEN,
    VID_END_TOKEN,
    VID_START_TOKEN,
    VID_TAG_TOKEN,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_tokenizer_for_sensevoice_glm4voice(tokenizer):
    token_list = [
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
        AUD_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        IMG_TAG_TOKEN,
        VID_TAG_TOKEN,
        AUD_TAG_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)

    token_list = [f"<|audio_{i}|>" for i in range(16384)]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=False)

    return tokenizer


class SenseVoiceGLM4VoiceTokenizer:
    def __init__(self, model_name_or_path, flow_path=None, rank=None):
        self.model_name_or_path = model_name_or_path
        self.flow_path = flow_path

        if rank is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            self.rank = rank % 8
        else:
            self.rank = rank
        logger.info(f"{self.rank=}")

        self.sample_rate = 16000

        self.is_discrete = True
        self.is_contiguous = True

        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}
        for sample_rate in [8000, 22050, 24000, 32000, 44100, 48000]:
            print(f"torchaudio.transforms.Resample {sample_rate=} {self.sample_rate=}")
            self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.sample_rate
            )

    def load_model(self):

        if hasattr(self, "whisper_model"):
            return
        if self.rank is not None:
            self.device = f"cuda:{self.rank}"
            torch.cuda.set_device(self.rank)
        else:
            self.device = "cpu"

        logger.info(f"{self.device=} Loading SenseVoiceSmall")
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id="FunAudioLLM/SenseVoiceSmall")
        
        _, self.kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=self.device)
        logger.info(f"{self.device=} Loading SenseVoiceSmall Done")

        logger.info(f"{self.device=} Loading GLM4VoiceTokenizer")
        self.whisper_model = (
            WhisperVQEncoder.from_pretrained(self.model_name_or_path).eval().to(self.device)
        )
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name_or_path)
        print(hasattr(self, "whisper_model"),'whisper_model')

        if self.flow_path is not None:
            flow_config = os.path.join(self.flow_path, "config.yaml")
            flow_checkpoint = os.path.join(self.flow_path, "flow.pt")
            hift_checkpoint = os.path.join(self.flow_path, "hift.pt")

            # Flow & Hift
            self.audio_decoder = AudioDecoder(
                config_path=flow_config,
                flow_ckpt_path=flow_checkpoint,
                hift_ckpt_path=hift_checkpoint,
                device=self.device,
            )
        logger.info(f"{self.device=} Loading GLM4VoiceTokenizer Done")

    def encode(self, audio_path, is_discrete=False, is_contiguous=True, **kwargs):

        import faulthandler
        faulthandler.enable()

        if not hasattr(self, "whisper_model"):
            self.load_model()

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete:
            audio_tokens = self.extract_speech_token(
                self.whisper_model, self.feature_extractor, [audio_path], device=self.device
            )[0]
            return audio_tokens

        if is_contiguous:

            audio, sample_rate = torchaudio.load(audio_path)
            audio = audio.mean(0)
            if sample_rate != self.sample_rate:
                if sample_rate not in self._resample_buffer:
                    print(f"torchaudio.transforms.Resample {sample_rate=} {self.sample_rate=} {self.device=}", flush=True)
                    self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=self.sample_rate
                    ).to(self.device)
                audio = audio.to(self.device)
                self._resample_buffer[sample_rate].to(self.device)
                audio = self._resample_buffer[sample_rate](audio[None, :])[0, :]
                audio = audio.cpu()

            frontend = self.kwargs["frontend"]

            speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)

            speech = speech[0]

            return speech

    def decode(self, audio_tokens, option_steps=10, **kwargs):
        if not hasattr(self, "whisper_model"):
            self.load_model()

        this_uuid = str(uuid.uuid4())
        this_uuid = "abc"

        tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)

        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)

        tts_speech, tts_mel = self.audio_decoder.token2wav(
            tts_token,
            uuid=this_uuid,
            prompt_token=flow_prompt_speech_token.to(self.device),
            prompt_feat=prompt_speech_feat.to(self.device),
            finalize=True,
            option_steps=option_steps,
        )
        tts_speechs = []
        tts_speechs.append(tts_speech.squeeze())
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        return tts_speech

    def apply_to_role(self, role, **kwargs):
        is_discrete = kwargs.get("is_discrete", False)
        if is_discrete and role in ["assistant", "gpt"]:
            return True

        is_contiguous = kwargs.get("is_contiguous", False)
        if is_contiguous and role in ["user", "human"]:
            return True

        return False

    def extract_speech_token(self, model, feature_extractor, utts, device="cuda"):
        with torch.no_grad():
            audios, indices = [], []
            for idx, utt in enumerate(utts):
                if isinstance(utt, tuple):
                    audio, sample_rate = utt
                else:
                    audio, sample_rate = torchaudio.load(utt)
                audio = audio.to(device)
                if sample_rate != 16000:
                    if sample_rate not in self._resample_buffer:
                        print(f"torchaudio.transforms.Resample {sample_rate=} {self.sample_rate=} {device=}")
                        self._resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                            orig_freq=sample_rate, new_freq=16000
                        ).to(device)
                    self._resample_buffer[sample_rate].to(device)
                    audio = self._resample_buffer[sample_rate](audio)

                audio = audio[0]
                audio = audio.cpu().numpy()
                time_step = 0
                while time_step * 16000 < audio.shape[0]:
                    audio_segment = audio[time_step * 16000 : (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(idx)
                    time_step += 30
            pooling_kernel_size = model.config.pooling_kernel_size or 1
            stride = (
                model.conv1.stride[0]
                * model.conv2.stride[0]
                * pooling_kernel_size
                * feature_extractor.hop_length
            )
            all_speech_tokens = [[] for _ in range(len(utts))]
            batch_size = 128
            for start in range(0, len(audios), batch_size):
                features = feature_extractor(
                    audios[start : start + batch_size],
                    sampling_rate=16000,
                    return_attention_mask=True,
                    return_tensors="pt",
                    device=device,
                    padding="longest",
                    pad_to_multiple_of=stride,
                )
                features = features.to(device=device)
                outputs = model(**features)
                speech_tokens = outputs.quantized_token_ids
                attention_mask = features.attention_mask[
                    :, :: model.conv1.stride[0] * model.conv2.stride[0]
                ]
                attention_mask = attention_mask[:, :: model.config.pooling_kernel_size]
                assert attention_mask.shape == speech_tokens.shape
                for i in range(len(speech_tokens)):
                    idx = indices[start + i]
                    speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                    all_speech_tokens[idx].extend(speech_token)

            return all_speech_tokens
