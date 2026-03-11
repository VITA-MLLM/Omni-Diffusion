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

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation import GenerationConfig

import torchaudio
from omni_diffusion.data.processor.audio_processor import add_audio_input_contiguous
from omni_diffusion.tokenizer import get_audio_tokenizer
from omni_diffusion.models.dream import DreamModel,DreamConfig,DreamTokenizer

def collate_fn(batches):
    return batches[0]

audio_tokenizer_type = "sensevoice_glm4voice"
audio_tokenizer_rank = 0
torch_dtype = torch.bfloat16
qwen2_chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""

class ASRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        audio_tokenizer,
        default_system_message=None,
        add_generation_prompt=True,
    ):
        data = load_dataset("json", data_files=json_path, keep_in_memory=False)
        self.data = data["train"]

        self.tokenizer = tokenizer
        self.add_generation_prompt = add_generation_prompt

        self.audio_tokenizer = audio_tokenizer
        self.default_system_message = default_system_message

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class S2SInference:
    def __init__(
        self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path=None, rank=None
    ):

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            chat_template=qwen2_chat_template,
        )

        model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).eval()
        print(f"{model.config.model_type=}")
        print(f"{model.hf_device_map=}")

        model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        model.generation_config.max_new_tokens = 8192
        model.generation_config.use_cache = True
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.num_beams = 1
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.chat_format = "chatml"
        model.generation_config.max_window_size = 8192
        model.generation_config.do_sample = False
        model.generation_config.top_k = 50
        print(f"{model.generation_config=}")

        audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
            flow_path=flow_path,
            rank=rank,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.add_generation_prompt = True
        self.default_system_message = []

    def run_infer(
        self,
        audio_path=None,
        stream_stride=4,
        max_returned_tokens=4096,
        sample_rate=16000,
        request_id="",
        audio_feats=None,
        message="",
        use_past=False,
        mode="luke",
        do_sample=False,
    ):

        AUD_TAG_TOKEN = "<|audio|>"
        AUD_CONTEXT_TOKEN = "<|context_of_audio|>"
        AUD_START_TOKEN = "<|begin_of_audio|>"
        AUD_END_TOKEN = "<|end_of_audio|>"

        system_message = self.default_system_message

        if audio_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        else:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        audio_paths = []
        if audio_path is not None:
            audio_paths.append(audio_path)
        input_ids, audios, audio_indices = add_audio_input_contiguous(
            input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
        )

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")
        
        print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

        self.model.generation_config.do_sample = do_sample

        max_new_tokens = int(input_ids.shape[-1] * 0.2)
        steps = int(max_new_tokens / 2)
        blocks = max_new_tokens
        
        print(max_new_tokens)
        print(steps)
        print(blocks)
        
        outputs, histories = self.model.generate(
            input_ids,
            audios=audios,
            audio_indices=audio_indices,
            temperature=0.0,
            top_p=0.9, 
            steps=steps,
            max_new_tokens = max_new_tokens,
            alg="entropy",
            block_size=blocks,
            tokenizer=self.tokenizer,
            task="ASR",
        )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        print(f"{output=}", flush=True)
    
        audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")

        audio_tokens = []
        text_tokens = []
        for token_id in outputs[0][input_ids.shape[-1]:].tolist():
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            else:
                text_tokens.append(token_id)

        try:
            text_tokens = text_tokens[:text_tokens.index(self.tokenizer.encode("<|im_end|>")[0])]
        except:
            text_tokens = text_tokens
        output = self.tokenizer.decode(text_tokens, skip_special_tokens=False)

        tts_speech = None

        return output, tts_speech

    
def inference(s2s_inference, dataloader, output_dir, rank):

    audio_offset = s2s_inference.tokenizer.convert_tokens_to_ids("<|audio_0|>")

    outputs = []
    output_path_ref = os.path.join(output_dir, f"ref_{rank}.txt")
    output_path_hyp = os.path.join(output_dir, f"hyp_{rank}.txt")


    for _, sample in enumerate(
        tqdm.tqdm(dataloader)
    ):
        output, tts_speech = s2s_inference.run_infer(
            audio_path=sample['audios'][0],
            message="Convert the speech to text.",
            mode=None,
        )
        hyp = output.replace("<|im_end|>", "").replace("<|endoftext|>", "").replace("-", "")
        ref = sample['messages'][1]['content']
        outputs.append((hyp, ref))

        print("")
        print("=" * 100)
        print(f"{hyp=}")
        print(f"{ref=}")

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_name_or_path", type=str, required=True, help="model_name_or_path")
    parser.add_argument("--flow_path", type=str, required=True, help="flow_path")
    parser.add_argument(
        "--audio_tokenizer_type", type=str, required=True, help="audio_tokenizer_type"
    )
    parser.add_argument(
        "--audio_tokenizer_path", type=str, required=True, help="audio_tokenizer_path"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")
    parser.add_argument("--json_path", type=str, required=True, help="json_path")

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

    # ================================================================
    print("Loading model")
    # device_map = "auto"
    device_map = "cuda"
    # torch_dtype=torch.float16
    torch_dtype = torch.bfloat16

    rank = torch.distributed.get_rank()

    s2s_inference = S2SInference(
        args.model_name_or_path, args.audio_tokenizer_path, args.audio_tokenizer_type, flow_path=args.flow_path, rank=rank
    )

    # ================================================================
    print("Loading data")
    dataset = ASRDataset(
        json_path=args.json_path,
        tokenizer=s2s_inference.tokenizer,
        audio_tokenizer=s2s_inference.audio_tokenizer,
        default_system_message=[],
        add_generation_prompt=True,
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
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = inference(s2s_inference, dataloader, args.output_dir, rank)

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
