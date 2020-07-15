#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=taskmaster-setup

# Preprocess, download, tokenization
# must point to transformers repo currently
#TRANSFORMERS_PATH=/home/js11133/transformers
#JIANT_PATH=/home/js11133/jiant/jiant
#WORKING_DIR=/scratch/js11133/jiant_working_dir
DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache

# Not supported: qa-srl, qamr
# span alignment tasks not yet supported
# TASKMASTER_TASKS=(boolq ccg cb commonsenseqa copa cosmosqa hellaswag mnli mrc qasrl qamr qqp record rte scitail socialiqa sst wic wsc)
TASKMASTER_TASKS=${1:-(boolq cb ccg commonsenseqa copa cosmosqa hellaswag mnli mrc qqp record rte scitail socialiqa sst wic)}
MODEL_TYPE=roberta-large

python ${JIANT_PATH}/proj/main/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${MODELS_DIR}/${MODEL_TYPE}

mkdir -p ${DATA_DIR}
for TASK_NAME in "${TASKMASTER_TASKS[@]}"
do
	python ${JIANT_PATH}/proj/main/tokenize_and_cache.py \
    		--task_config_path ${DATA_DIR}/configs/${TASK_NAME}.json \
    		--model_type ${MODEL_TYPE} \
    		--model_tokenizer_path ${MODELS_DIR}/${MODEL_TYPE}/tokenizer \
    		--phases train,val,test \
    		--max_seq_length 256 \
    		--do_iter \
    		--smart_truncate \
    		--output_dir ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME}
done
