import argparse
import itertools
import json
import os
import random
import re
import sys
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import tqdm
from datasets import load_dataset
from tn.english.normalizer import Normalizer as EnNormalizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation import GenerationConfig

import torchaudio
from omni_diffusion.tokenizer import get_audio_tokenizer
from omni_diffusion.models.dream import DreamModel,DreamConfig,DreamTokenizer
from evaluate_asr import InferenceSampler


qwen2_chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""


def collate_fn(batches):
    input_ids = [sample["input_ids"] for sample in batches]

    refs = [sample["ref"] for sample in batches]
    filenames = [sample["filename"] for sample in batches]

    return input_ids, refs, filenames


class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, audio_tokenizer, default_system_message=None, 
                 add_generation_prompt=True):
        data = load_dataset("json", data_files=json_path, keep_in_memory=False)
        self.data = data["train"]

        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.default_system_message = default_system_message
        self.add_generation_prompt = add_generation_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        messages = []

        if self.default_system_message is not None:
            messages = self.default_system_message + messages

        role = "user"
        content = sample["messages"][0]["content"]
        messages.append(
            {
                "role": role,
                "content": content,
            }
        )
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
            return_tensors="pt",
            enable_thinking=False,
        )

        ref = sample["messages"][0]["content"]
        ref = ref.replace("Convert the text to speech.\n", "")
        ref = ref.strip()

        filepath = sample["audios"][0]
        filename = os.path.basename(filepath)

        return {
            "input_ids": input_ids,
            "ref": ref,
            "filename": filename,
        }


def inference(model, tokenizer, audio_tokenizer, dataloader, output_dir, asr_model, alg, step_ratio=2):

    audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    en_tn_model = EnNormalizer(overwrite_cache=True)

    outputs = []

    for _, (
        batched_input_ids,
        batched_ref,
        batched_filename,
    ) in enumerate(tqdm.tqdm(dataloader)):
        for input_ids, ref, filename in zip(
            batched_input_ids, batched_ref, batched_filename
        ):
            max_new_tokens = int(len(tokenizer.encode(ref)) * 3.5)
            steps = max_new_tokens // step_ratio

            responses = model.generate(
                input_ids.to(model.device),
                temperature=0,
                top_p=0.9,
                steps=steps,
                max_new_tokens = max_new_tokens,
                alg=alg,
                tokenizer=tokenizer,
                repeat_penalty=1.2,
            )

            response = responses[0][0][len(input_ids[0]) :]

            text_tokens = []
            audio_tokens = []
            for token_id in response:
                if token_id >= audio_offset and token_id < audio_offset + 16384:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)

            if len(audio_tokens) == 0:
                continue

            tts_speech = audio_tokenizer.decode(audio_tokens)
            wav_dir = os.path.join(output_dir, "audio")
            wav_path = os.path.join(wav_dir, filename + ".wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

            hyp = asr_model(wav_path, return_timestamps=True)["text"].strip()

            hyp = en_tn_model.normalize(hyp)
            ref = en_tn_model.normalize(ref)

            hyp = re.sub(r"\W+", " ", hyp)
            ref = re.sub(r"\W+", " ", ref)

            outputs.append((hyp, ref))

            print(f"{hyp=}")
            print(f"{ref=}")
            print("=" * 100)
            print(f"{tokenizer.decode(response, skip_special_tokens=False)}")
            print(f"{filename=}")

    return outputs


def load_asr_model():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    rank = torch.distributed.get_rank()
    device = f"cuda:{rank}"
    torch_dtype = torch.float16

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_name_or_path", type=str, required=True, help="model_name_or_path")
    parser.add_argument(
        "--audio_tokenizer_path", type=str, required=True, help="audio_tokenizer_path"
    )
    parser.add_argument(
        "--audio_tokenizer_type", type=str, required=True, help="audio_tokenizer_type"
    )
    parser.add_argument("--flow_path", type=str, required=True, help="flow_path")

    parser.add_argument("--json_path", type=str, required=True, help="json_path")
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--speaker_prompt", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    print(f"{args=}")

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
        timeout=timedelta(seconds=7200),
    )

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    random.seed(42)
    torch.manual_seed(42)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    # ================================================================
    add_generation_prompt = True
    default_system_message = []

    # ================================================================
    print("Loading model")
    device = "cuda"
    # device_map = "auto"
    device_map = "cuda"
    # torch_dtype=torch.float16
    torch_dtype = torch.bfloat16

    rank = torch.distributed.get_rank()

    audio_tokenizer = get_audio_tokenizer(
        args.audio_tokenizer_path, args.audio_tokenizer_type, flow_path=args.flow_path, rank=rank
    )

    chat_template = qwen2_chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        chat_template=chat_template,
    )
    # print("tokenizer", tokenizer)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).eval()
    # print("model", model)

    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    model.generation_config.max_new_tokens = 4096
    model.generation_config.chat_format = "chatml"
    model.generation_config.max_window_size = 8192
    model.generation_config.use_cache = True
    model.generation_config.do_sample = True
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    asr_model = load_asr_model()

    # ================================================================
    print("Loading data")
    dataset = TTSDataset(
        json_path=args.json_path,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        default_system_message=default_system_message,
        add_generation_prompt=add_generation_prompt,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(
            collate_fn,
        ),
    )

    # ================================================================
    outputs = inference(model, tokenizer, audio_tokenizer, dataloader, args.output_dir, asr_model, "entropy-penalty")

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        # json_name = Path("_".join(os.path.normpath(args.json_path).split(os.sep)[-2:])).stem
        json_name = Path(os.path.normpath(args.json_path).split(os.sep)[-1]).stem
        hyp_path = os.path.join(args.output_dir, f"{json_name}_hyp.txt")
        ref_path = os.path.join(args.output_dir, f"{json_name}_ref.txt")

        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        os.makedirs(os.path.dirname(hyp_path), exist_ok=True)

        hyp_file = open(hyp_path, "w")
        ref_file = open(ref_path, "w")

        for sample_idx, (hyp, ref) in enumerate(merged_outputs):
            hyp_file.write(f"{sample_idx} {hyp}" + "\n")
            ref_file.write(f"{sample_idx} {ref}" + "\n")

        hyp_file.close()
        ref_file.close()

        hyp_ref_path = os.path.join(args.output_dir, f"{json_name}_hyp_ref.json")
        hyp_ref_file = open(hyp_ref_path, "w")
        json.dump(merged_outputs, hyp_ref_file, indent=4)
        hyp_ref_file.close()

    torch.distributed.barrier()
    print("Done.")
