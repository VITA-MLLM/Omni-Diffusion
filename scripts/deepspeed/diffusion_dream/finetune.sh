#!/bin/bash

set -e
set -x

SEQ_LENGTH="$1"
if [ -z "$SEQ_LENGTH" ]
then
    SEQ_LENGTH=1024
fi

timestamp="$2"
if [ -z "$timestamp" ]
then
    timestamp=`date +'%Y%m%d_%H'`0000
fi

######################################################################
export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Omni-Diffusion/

cd ${CODE_PATH}

######################################################################
source ${CODE_PATH}/scripts/set_env_ds_gpu.sh
python -m pip install diffusers==0.32.2
python -m pip install jaxtyping
python -m pip install peft==0.17.1

######################################################################
OUTPUT_DIR=${ROOT_PATH}/output/LM/"$0"/${timestamp}/

mkdir -p ${OUTPUT_DIR}
rsync -avh $0 ${OUTPUT_DIR}

export HF_HOME="${ROOT_PATH}/data/HF_HOME_node${INDEX}/"
mkdir -p ${HF_HOME}

export TRITON_CACHE_DIR=${CODE_PATH}

export PYTHONPATH=$PYTHONPATH:${CODE_PATH}/third_party/GLM-4-Voice:${CODE_PATH}/third_party/GLM-4-Voice/third_party/Matcha-TTS/

######################################################################
LOG=${OUTPUT_DIR}/log_node${INDEX}.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo ${@}

######################################################################
DATA_PATH=${CODE_PATH}/configs/finetune.yaml

MODEL_NAME_OR_PATH="../models/Omni-Diffusion"
AUDIO_TOKENIZER_PATH="../models/THUDM/glm-4-voice-tokenizer"
AUDIO_MODEL_NAME_OR_PATH="../models/FunAudioLLM/SenseVoiceSmall/model.pt"
IMAGE_TOKENIZER_PATH="../models/magvitv2"

rsync -avh ${DATA_PATH} ${OUTPUT_DIR}

######################################################################
export NCCL_NVLS_ENABLE=0

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port 45788
"

# torchrun $DISTRIBUTED_ARGS tools/finetune_dream_v4_51_3.py \
python -m torch.distributed.run $DISTRIBUTED_ARGS tools/finetune_dream_v4_51_3.py \
    --log_level "info" \
    --do_train \
    --config_name ${CODE_PATH}/omni_diffusion/models/dream/config_dream_resume.json \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_model_name_or_path ${AUDIO_MODEL_NAME_OR_PATH} \
    --audio_tokenizer_path $AUDIO_TOKENIZER_PATH \
    --audio_tokenizer_type "sensevoice_glm4voice" \
    --image_tokenizer_path $IMAGE_TOKENIZER_PATH \
    --dataset_name $DATA_PATH \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --max_steps 24000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 600 \
    --save_total_limit 100 \
    --learning_rate 1.00e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length ${SEQ_LENGTH} \
    --gradient_checkpointing False \
    --deepspeed ${CODE_PATH}/scripts/deepspeed/ds_config_zero2.json \
    --trust_remote_code True \
    --ddp_timeout 7200 \
    --ddp_backend ${DISTRIBUTED_BACKEND} \
    --attn_implementation flash_attention_2 \
    --seed 956 \
    --data_seed 956 \
    --reset_attention_mask \
    --reset_position_ids \
    --dataloader_num_workers 1 \
    --audio-model-freeze \
    --image_size 256 \
    --overwrite_output_dir \

set +x
