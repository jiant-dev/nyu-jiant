# Preprocess, download, tokenization
# must point to transformers repo currently
#TRANSFORMERS_PATH=/home/js11133/transformers
#JIANT_PATH=/home/js11133/jiant/jiant
#WORKING_DIR=/scratch/js11133/jiant_working_dir
DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache

# ERROR with wnli, diagnostic, snli
#GLUE_TASKS=(cola mrpc qqp sst stsb mnli qnli rte)
GLUE_TASKS=(snli qnli)
MODEL_TYPE=bert-base-cased

mkdir -p ${DATA_DIR}
python ${TRANSFORMERS_PATH}/utils/download_glue_data.py \
    --data_dir ${DATA_DIR}/raw_glue_data

python ${JIANT_PATH}/scripts/preproc/export_glue_data.py \
    --input_base_path ${DATA_DIR}/raw_glue_data \
    --output_base_path ${DATA_DIR}/glue

python ${JIANT_PATH}/scripts/preproc/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${MODELS_DIR}/${MODEL_TYPE}

for TASK_NAME in "${GLUE_TASKS[@]}"
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
