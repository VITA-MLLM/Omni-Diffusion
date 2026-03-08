#!/bin/bash

set -e
set -x

timestamp=test_libritts

######################################################################
export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Omni-Diffusion/

######################################################################
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${CODE_PATH}/scripts/set_env_ds_gpu.sh

######################################################################
OUTPUT_DIR=${ROOT_PATH}/outputs/LM/"$0"/${timestamp}/

mkdir -p ${OUTPUT_DIR}
rsync -avh $0 ${OUTPUT_DIR}

export HF_HOME="${ROOT_PATH}/data/HF_HOME/"
mkdir -p ${HF_HOME}
export HF_ENDPOINT=https://hf-mirror.com

export MODELSCOPE_CACHE="${ROOT_PATH}/data/MODELSCOPE_CACHE/"
mkdir -p ${MODELSCOPE_CACHE}

export LC_ALL="en_US.utf8"

######################################################################
LOG=${OUTPUT_DIR}/log_node${INDEX}.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

apt-get update && apt install -y ffmpeg

pip install jiwer==3.1.0

MODEL_NAME_OR_PATH=$1
AUDIO_TOKENIZER_PATH=$2
FLOW_PATH=$3

AUDIO_TOKENIZER_TYPE="sensevoice_glm4voice"
export PYTHONPATH=${PYTHONPATH}:${CODE_PATH}/third_party/GLM-4-Voice/:${CODE_PATH}/third_party/GLM-4-Voice/cosyvoice/:${CODE_PATH}/third_party/GLM-4-Voice/third_party/Matcha-TTS/

######################################################################
export NPROC_PER_NODE=1
export NNODES=1
export NODE_RANK=0
export MASTER_PORT=45678
export MASTER_ADDR=127.0.0.1
DISTRIBUTED_ARGS="
--nproc_per_node $NPROC_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT
"

######################################################################
JSON_PATH=/data/jsonl/mythicinfinity/libritts/test.clean.jsonl

torchrun ${DISTRIBUTED_ARGS} tools/evaluate_libritts.py \
	--json_path ${JSON_PATH} \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
	--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
	--flow_path ${FLOW_PATH} \
	--output_dir ${OUTPUT_DIR}/libritts/ \


python tools/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/libritts/test.clean_ref.txt ${OUTPUT_DIR}/libritts/test.clean_hyp.txt
echo "copypaste WER: ${JSON_PATH}"

set +x
