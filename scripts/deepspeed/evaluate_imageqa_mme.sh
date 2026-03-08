#!/bin/bash

set -e
set -x

timestamp=test_mme

######################################################################
export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/Omni-Diffusion/

######################################################################
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${CODE_PATH}/scripts/set_env_ds_gpu.sh
#pip3 install transformers==4.48.3

######################################################################
OUTPUT_DIR=${CODE_PATH}/outputs/"$0"/${timestamp}/

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


######################################################################
MODEL_NAME_OR_PATH=$1
IMAGE_TOKENIZER_PATH=$2
MME_DIR=$3

export PYTHONPATH=${PYTHONPATH}:${CODE_PATH}/third_party/GLM-4-Voice/:${CODE_PATH}/third_party/GLM-4-Voice/cosyvoice/:${CODE_PATH}/third_party/GLM-4-Voice/third_party/Matcha-TTS/

python tools/evaluate_imageqa_mme.py \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--image_tokenizer_path ${IMAGE_TOKENIZER_PATH} \
	--output_dir ${OUTPUT_DIR}/mme/ \
	--mme_dir ${MME_DIR} \


python ${MME_DIR}/eval_tool/calculation.py --results_dir ${OUTPUT_DIR}/mme/

set +x
