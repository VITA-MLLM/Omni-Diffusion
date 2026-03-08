#set -e
#set -x

######################################################################

export DISTRIBUTED_BACKEND="nccl"
export CUDA_DEVICE_MAX_CONNECTIONS=1

######################################################################
python -m pip install -r requirements_ds_gpu.txt
python -m pip install -e `pwd`

######################################################################

export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export MASTER_PORT=45678

if [ -z "$NPROC_PER_NODE" ]
then
    export NPROC_PER_NODE=8
    export NNODES=1
    export NODE_RANK=0
    export MASTER_ADDR=127.0.0.1
fi

######################################################################
