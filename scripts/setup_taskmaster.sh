# Preprocess, download, tokenization
# must point to transformers repo currently
#TRANSFORMERS_PATH=/home/js11133/transformers
#JIANT_PATH=/home/js11133/jiant/jiant
#WORKING_DIR=/scratch/js11133/jiant_working_dir
DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache

# Not supported: qa-srl, qamr
# Where is commonsenseqa data?
# TASKMASTER_TASKS=(boolq ccg commitmentbank commonsenseqa copa cosmosqa hellaswag mnli multirc qqp record rte scitail socialqa sst wic wsc)
TASKMASTER_TASKS=(boolq ccg commitmentbank copa cosmosqa hellaswag mnli multirc qqp record rte socialqa sst wic wsc)
MODEL_TYPE=roberta-large

python ${JIANT_PATH}/scripts/preproc/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${MODELS_DIR}/${MODEL_TYPE}

mkdir -p ${DATA_DIR}
for TASK_NAME in "${TASKMASTER_TASKS[@]}"
do
	python ${JIANT_PATH}/proj/simple/tokenize_and_cache.py \
    		--task_config_path ${DATA_DIR}/glue/configs/${TASK_NAME}.json \
    		--model_type ${MODEL_TYPE} \
    		--model_tokenizer_path ${MODELS_DIR}/${MODEL_TYPE}/tokenizer \
    		--phases train,val,test \
    		--max_seq_length 256 \
    		--do_iter \
    		--smart_truncate \
    		--output_dir ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME}
done
