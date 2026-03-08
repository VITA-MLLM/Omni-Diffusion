import json
import logging
import os
import random
import re
import sys
import time
import uuid
from threading import Thread
from typing import Optional

import torch
import tqdm
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig

import torchaudio
from omni_diffusion.data.processor.audio_processor import add_audio_input_contiguous
from omni_diffusion.tokenizer import get_audio_tokenizer
from omni_diffusion.models.dream import DreamModel,DreamConfig,DreamTokenizer
from omni_diffusion.data.processor.image_processor import ImageProcessor
import cv2
import random
import numpy as np
import torchaudio
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append("third_party/GLM-4-Voice/")
sys.path.append("third_party/GLM-4-Voice/cosyvoice/")
sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")

qwen2_chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""

def set_seed(seed: int = 42):
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch（CPU）
    torch.manual_seed(seed)

    # 4. PyTorch（GPU）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

def find_audio_segments_regex(text):
    """
    Find all substrings between <|begin_of_audio|> and <|end_of_audio|> using regex.

    Args:
        text (str): The input string to search through

    Returns:
        list: A list of all found audio segments (substrings between the delimiters)
    """
    pattern = re.compile(r"<\|begin_of_audio\|>(.*?)<\|end_of_audio\|>", re.DOTALL)
    segments = pattern.findall(text)
    return [segment.strip() for segment in segments]


def extract_token_ids_as_int(text):
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    token_ids = pattern.findall(text)
    return [int(id) for id in token_ids]


class S2SInference:
    def __init__(
        self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, image_tokenizer_path, flow_path=None,
    ):

        config = DreamConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        chat_template = qwen2_chat_template
        add_generation_prompt = True
        default_system_message = []

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )
        # print(f"{tokenizer=}")
        print(f"{tokenizer.get_chat_template()=}")

        model = DreamModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).eval()
        # print("model", model)
        print(f"{model.config.model_type=}")
        print(f"{model.hf_device_map=}")

        model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        model.generation_config.max_new_tokens = 8192
        model.generation_config.chat_format = "chatml"
        model.generation_config.max_window_size = 8192
        model.generation_config.use_cache = True
        # model.generation_config.use_cache = False
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_k = 50
        model.generation_config.top_p = 1.0
        model.generation_config.num_beams = 1
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        print(f"{model.generation_config=}")

        audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
            flow_path=flow_path,
            rank=audio_tokenizer_rank,
        )

        image_processor = ImageProcessor(
            image_tokenizer_path,
            'dynamic',
            image_size=512,
            normalize_type='imagenet',
            min_patch_grid=1,
            max_patch_grid=12,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.add_generation_prompt = add_generation_prompt
        self.default_system_message = default_system_message
        self.image_processor = image_processor
        self.image_processor.image_tokenizer.rank = 0
        self.image_processor.load_model()

        audio_0_id = tokenizer("<|audio_0|>").input_ids[0]
        print(f"{audio_0_id=}")

    def run_infer(
        self,
        audio_path=None,
        image_path=None,
        max_tokens=128,
        steps=128,
        message="",
        add_boa_token=0,
        task = "",
        alg="entropy",
        cfg=0.0,
        max_position_penalty=1.0,
        repeat_penalty=1.0,
        output_text_only=False,
    ):

        AUD_TAG_TOKEN = "<|audio|>"
        AUD_CONTEXT_TOKEN = "<|context_of_audio|>"
        AUD_START_TOKEN = "<|begin_of_audio|>"
        AUD_END_TOKEN = "<|end_of_audio|>"

        system_message = self.default_system_message

        if task == "S2I":
            system_message = [
                {
                    "role": "system",
                    "content": f"Please generate an image based on the input audio.",
                },
            ]
        if task == "SQA":
            system_message = [
                {
                    "role": "system",
                    "content": f"Please response the input audio.",
                },
            ]
        if task == "SVQA":
            system_message = [
                {
                    "role": "system",
                    "content": f"Please response the input audio based on the given image.",
                },
            ]

        if audio_path is not None and image_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": "<|audio|>\n<|image|>",
                },
            ]
        elif audio_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        elif image_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|image|>",
                },
            ]
        else:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )
        
        if image_path is not None:
            image_tokens = self.image_processor.process_images_with_subpatch(image_path, 512)
            image_tokens = self.image_processor.get_image_token(image_tokens)
            image_tokens = image_tokens[0].tolist()
            image_tokens = "".join(f"<|image_{i}|>" for i in image_tokens)

            IMG_START_TOKEN = "<|begin_of_image|>"
            IMG_END_TOKEN = "<|end_of_image|>"

            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|image|>", f"{IMG_START_TOKEN}{image_tokens}{IMG_END_TOKEN}"
            )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        if audio_path is not None and self.audio_tokenizer.apply_to_role(
            "user", is_contiguous=True
        ):
            # contiguous codec
            audio_paths = []
            if audio_path is not None:
                audio_paths.append(audio_path)
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")
        
        print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)
        outputs, histories = self.model.generate(
            input_ids,
            audios=audios,
            audio_indices=audio_indices,
            temperature=0.0,
            top_p=0.9,  
            steps=steps,
            max_new_tokens = max_tokens,
            alg=alg,
            cfg=cfg,
            tokenizer=self.tokenizer,
            add_boa_token=add_boa_token,
            max_position_penalty=max_position_penalty,
            repeat_penalty=repeat_penalty,
            output_text_only=output_text_only,
            task=task,
        )

        output = self.tokenizer.decode(outputs[0][input_ids.shape[1]: ], skip_special_tokens=False)
        print(f"{output=}", flush=True)
        
        audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
        audio_tokens = []
        image_offset = self.tokenizer.convert_tokens_to_ids("<|image_0|>")
        image_tokens = []
        text_tokens = []
        for token_id in outputs[0][input_ids.shape[1]: ]:
            if token_id >= audio_offset and token_id < audio_offset + 16384:
                audio_tokens.append(token_id - audio_offset)
            elif token_id >= image_offset:
                image_tokens.append(token_id - image_offset)
            else:
                text_tokens.append(token_id)

        if len(audio_tokens) > 0:
            tts_speech = self.audio_tokenizer.decode(
                audio_tokens, source_speech_16k=None
            )
        else:
            tts_speech = None

        if len(image_tokens) < 256 and len(image_tokens) > 0:
            image_tokens += [image_tokens[-1]] * (256 - len(image_tokens))

        image = None
        if len(image_tokens) > 0:
            gen_token_ids = torch.stack(image_tokens, dim=0).unsqueeze(0)
            gen_token_ids = torch.clamp(gen_token_ids, max=8192 - 1, min=0)
            image = self.image_processor.image_tokenizer.image_tokenizer.decode_code(gen_token_ids[:, :256]) 
            image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
            image *= 255.0
            image = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            image = image[:, :, :, [2, 1, 0]][0]

        output = self.tokenizer.decode(text_tokens, skip_special_tokens=True)

        return output, tts_speech, image


def tts_task(s2s_inference, max_tokens, steps, alg, repeat_penalty):
    TTS_texts = [
        "Get the trust fund to the bank early.",
        "It's a nice walk though, with trees and a river and stuff.",
    ]
    outputs = []
    audios = []

    for text in TTS_texts:
        print("=" * 100)
        print("tts_task")
        print(f"{text=}")

        output, tts_speech, images = s2s_inference.run_infer(
            message="Convert the text to speech.\n" + text,
            task="TTS",
            max_tokens=max_tokens,
            steps=steps,
            alg=alg,
            repeat_penalty=repeat_penalty,
        )
        outputs.append(output)
        audios.append(tts_speech)

    return outputs, audios


def asr_task(s2s_inference, max_tokens, steps, alg, repeat_penalty):
    outputs = []
    for audio_path in [
        "asset/asr_0.wav", 
    ]:
        print("=" * 100)
        print("asr_task")
        print(f"{audio_path=}")

        output, tts_speech, images = s2s_inference.run_infer(
            audio_path=audio_path,
            message="Convert the speech to text.",
            task="ASR",
            max_tokens=max_tokens,
            steps=steps,
            alg=alg,
            repeat_penalty=repeat_penalty,
        )
        print(f"{output=}", flush=True)
        outputs.append(output)

    return outputs


def vqa_task(s2s_inference, max_tokens, steps, alg, repeat_penalty):
    outputs = []
    image_path_list = [
        "asset/vqa_0.png",
    ]
    text_query_list = [
        "Is the glass of orange juice half empty or half full?",
    ]
    for image_path, text in zip(image_path_list, text_query_list):
        output, _, _ = s2s_inference.run_infer(
            image_path=image_path,
            message=text,
            task="VQA",
            max_tokens=max_tokens,
            steps=steps,
            alg=alg,
            repeat_penalty=repeat_penalty,
        )
        print(f"{output=}", flush=True)
        outputs.append(output)

    return outputs


def t2i_task(s2s_inference, max_tokens, steps, alg, repeat_penalty):
    outputs = []
    images = []
    
    prompts = [
        "The image shows a landscape background with double exposure glasses of wine, displaying a hyperealistic and detailed view of the subject.",
    ]
    for prompt in prompts:
        output, _, image = s2s_inference.run_infer(
            message="Generate an image based on the provided text description.\n" + prompt,
            task="T2I",
            max_tokens=max_tokens,
            steps=steps,
            alg=alg,
            repeat_penalty=repeat_penalty,
            max_position_penalty=2.0,
        )
        print(f"{output=}", flush=True)
        outputs.append(output)
        images.append(image)

    return outputs, images

def s2i_task(s2s_inference, max_tokens, steps, alg, repeat_penalty):
    outputs = []
    images = []
    for audio_path in [
        "asset/s2i_0.wav", #A super realistic and hyper-detailed 8k image showing a fantasy night scene with an amazing beach under the full moon, lit by dramatic lighting.  
        ]:

        output, _, image = s2s_inference.run_infer(
            audio_path=audio_path,
            message="",
            task="S2I",
            max_tokens=max_tokens,
            steps=steps,
            alg=alg,
            repeat_penalty=repeat_penalty,
            max_position_penalty=2.0,
            cfg=2.0,
        )
        print(f"{output=}", flush=True)
        outputs.append(output)
        images.append(image)

    return outputs, images


def svqa_task(s2s_inference, max_tokens, steps, alg, repeat_penalty):
    outputs = []
    audio_paths = [
        "asset/svqa_0.wav", # What game is this?
    ]
    image_paths = [
        "asset/svqa_0.jpg",
    ]
    
    outputs = []
    audios = []
    for audio_path, img_path in zip(audio_paths, image_paths):
        output, tts_speech, image = s2s_inference.run_infer(
            audio_path=audio_path,
            image_path=img_path,
            message="",
            add_boa_token=True,
            max_tokens=max_tokens,
            steps=steps,
            alg=alg,
            repeat_penalty=repeat_penalty,
            task="SVQA",
        )
        print(f"{output=}", flush=True)
        outputs.append(output)
        audios.append(tts_speech)

    return outputs, audios

def save_output(output_path, prefix, output, images, speech):
    if images is not None:
        os.makedirs(f"{output_path}/images/", exist_ok=True)
        for index, img in enumerate(images):
            if img is not None:
                cv2.imwrite(f"{output_path}/images/{prefix}_{index}.png", img)

    if speech is not None:
        os.makedirs(f"{output_path}/wavs/", exist_ok=True)
        for index, audio in enumerate(speech):
            if audio is not None:
                torchaudio.save(f"{output_path}/wavs/{prefix}_{index}.wav", audio.unsqueeze(0), 22050, format="wav")

    log_path = os.path.join(output_path, f"{prefix}_log.txt")
    with open(log_path, "w") as fout:
        print("\n\n".join(output), file=fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_name_or_path", type=str, required=True, help="model_name_or_path")
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")
    parser.add_argument(
        "--audio_tokenizer_path", type=str, default="THUDM/glm-4-voice-tokenizer"
    )
    parser.add_argument(
        "--image_tokenizer_path", type=str, default="showlab/magvitv2"
    )
    parser.add_argument("--flow_path", type=str, default="THUDM/glm-4-voice-decoder")

    args = parser.parse_args()

    device_map = "cuda:0"
    audio_tokenizer_rank = 0
    torch_dtype = torch.bfloat16
    output_path = args.output_dir
    audio_tokenizer_path = args.audio_tokenizer_path
    image_tokenizer_path = args.image_tokenizer_path
    flow_path = args.flow_path
    model_name_or_path = args.model_name_or_path
    audio_tokenizer_type = "sensevoice_glm4voice"

    os.makedirs(output_path, exist_ok=True)

    s2s_inference = S2SInference(
        model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, image_tokenizer_path, flow_path=flow_path,
    )
    
    images = None
    speech = None

    # speech-to-image
    output, images = s2i_task(s2s_inference, 260, 260, "entropy-penalty", 1.5)
    save_output(output_path, "s2i", output, images, None)

    # text-to-image
    output, images = t2i_task(s2s_inference, 260, 260, "entropy-penalty", 1.2)
    save_output(output_path, "t2i", output, images, None)

    # spoken visual qa
    output, speech = svqa_task(s2s_inference, 128, 128, "entropy-penalty", 1.2)
    save_output(output_path, "svqa", output, None, speech)

    # visual qa
    output = vqa_task(s2s_inference, 64, 64, "entropy", 1.0)
    save_output(output_path, "vqa", output, None, None)

    # tts
    output, speech = tts_task(s2s_inference, 50, 25, "entropy", 1.0)
    save_output(output_path, "tts", output, None, speech)

    # asr
    output = asr_task(s2s_inference, 50, 25, "entropy", 1.0)
    save_output(output_path, "asr", output, None, None)
