# Manual runner example

```bash
WORKING_DIR=...  # Choose a working dir

DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache

for TASK_NAME in mnli qnli rte
do
    python jiant/proj/simple/tokenize_and_cache.py \
        --task_config_path ${DATA_DIR}/glue/configs/${TASK_NAME}.json \
        --model_type ${TASK_NAME} \
        --model_tokenizer_path ${MODELS_DIR}/roberta-base/tokenizer \
        --phases train,val,test \
        --max_seq_length 256 \
        --do_iter \
        --smart_truncate \
        --output_dir ${WORKING_DIR}/cache/${TASK_NAME}
done
python examples/create_model.py \
    --model_config_path ${MODELS_DIR}/roberta-base/model/roberta-base.json \
    --model_tokenizer_path ${MODELS_DIR}/roberta-base/tokenizer \
    --task_config_base_path ${DATA_DIR}/glue/configs/ \
    --task_cache_base_path ${WORKING_DIR}/cache
```