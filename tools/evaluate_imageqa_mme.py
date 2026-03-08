import argparse
import itertools
import json
import os
import random
import sys
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig

import torchaudio
from omni_diffusion.data.processor.audio_processor import add_audio_input_contiguous
from omni_diffusion.tokenizer import get_audio_tokenizer
from omni_diffusion.models.dream import DreamModel,DreamConfig,DreamTokenizer
from omni_diffusion.data.processor.image_processor import ImageProcessor
import cv2

device_map = "cuda:0"
audio_tokenizer_rank = 0
torch_dtype = torch.bfloat16


sys.path.append("third_party/GLM-4-Voice/")
sys.path.append("third_party/GLM-4-Voice/cosyvoice/")
sys.path.append("third_party/GLM-4-Voice/third_party/Matcha-TTS/")

audio_tokenizer_type = "sensevoice_glm4voice"

qwen2_chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""


class S2SInference:
    def __init__(
        self, model_name_or_path, image_tokenizer_path
    ):

        config = DreamConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            chat_template=qwen2_chat_template,
        )

        model = DreamModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).eval()

        model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        model.generation_config.chat_format = "chatml"
        model.generation_config.max_new_tokens = 8192
        model.generation_config.max_window_size = 8192
        model.generation_config.temperature = 1.0
        model.generation_config.top_k = 50
        model.generation_config.top_p = 1.0
        model.generation_config.num_beams = 1
        model.generation_config.use_cache = True
        model.generation_config.do_sample = False
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        print(f"{model.generation_config=}")

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
        self.image_processor = image_processor
        self.image_processor.image_tokenizer.rank = 0
        self.image_processor.load_model()
        self.add_generation_prompt = True
        self.default_system_message = []
        

    def run_infer(
        self,
        audio_path=None,
        image_path=None,
        message="",
        mode="luke",
        do_sample=False,
        steps = 64,
        max_new_tokens = 64,
        cfg=0.0,
        alg="entropy",
        repeat_penalty=1.0,
    ):

        AUD_TAG_TOKEN = "<|audio|>"
        AUD_CONTEXT_TOKEN = "<|context_of_audio|>"
        AUD_START_TOKEN = "<|begin_of_audio|>"
        AUD_END_TOKEN = "<|end_of_audio|>"

        system_message = self.default_system_message

        assert "<|image|>" in message
        messages = system_message + [
            {
                "role": "user",
                "content": message,
            },
        ]

        if image_path is not None:
            image_tokens = self.image_processor.process_images_with_subpatch(image_path)
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
        audios = None
        audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")
        
        print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

        self.model.generation_config.do_sample = do_sample
        
        print(max_new_tokens)
        print(steps)
        print(cfg)
        outputs, histories = self.model.generate(
            input_ids,
            audios=audios,
            audio_indices=audio_indices,
            temperature=0.0,
            top_p=0.9,  
            steps=steps,
            max_new_tokens = max_new_tokens,
            alg=alg,
            tokenizer=self.tokenizer,
            cfg=cfg,
            repeat_penalty=repeat_penalty,
            task="VQA",
        )
        
        output = self.tokenizer.decode(outputs[0][input_ids.shape[-1]: ], skip_special_tokens=False)
        print(f"{output=}", flush=True)

        return output

    

def inference(s2s_inference, mme_dir, output_dir, rank, ):

    audio_offset = s2s_inference.tokenizer.convert_tokens_to_ids("<|audio_0|>")

    outputs = []

    _, _, test_names = next(os.walk(os.path.join(mme_dir, "eval_tool/Your_Results")))
    for test_name in test_names:
        with open(os.path.join(mme_dir, "eval_tool/Your_Results", test_name)) as f: datas = f.readlines()
        outputs_list = []
        output_path = os.path.join(output_dir, test_name)
        
        with open(output_path, "w") as fout:
            for index, data in enumerate(datas):
                data = data.replace("\n", "").split("\t")
                img_name = data[0]
                ques = data[1]
                ans = data[2]

                img_path = os.path.join(mme_dir, "images", test_name.replace(".txt", ""), img_name)

                output = s2s_inference.run_infer(
                    image_path=img_path,
                    message=ques + "\n<|image|>",
                    mode=None,
                    max_new_tokens=8,
                    steps=8,
                    alg="entropy",
                )

                outputs_list.append(
                    {
                        "image_id": img_name, "ques": ques, "pred": output.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("\n", ""), "gt": ans
                    }
                )
                fout.write(
                    f"{img_name}\t{ques}\t{ans}\t{output.replace('<|im_end|>', '').replace('<|endoftext|>', '').replace('\n', '')}\n"
                )
                fout.flush()

    return outputs_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_name_or_path", type=str, required=True, help="model_name_or_path")
    parser.add_argument(
        "--image_tokenizer_path", type=str, required=True, help="image_tokenizer_path"
    )

    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")
    parser.add_argument("--mme_dir", type=str, required=True, help="mme_dir")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    print(f"{args=}")

    random.seed(42)
    torch.manual_seed(42)

    s2s_inference = S2SInference(
        args.model_name_or_path, args.image_tokenizer_path
    )

    # ================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = inference(s2s_inference, args.mme_dir, args.output_dir, 0)

    print("Done.")
