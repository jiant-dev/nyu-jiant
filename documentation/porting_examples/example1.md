# Preprocess, download, tokenization

```bash
TRANSFORMERS_PATH=...  # wherever your installation/repo of Transformers is
WORKING_DIR=...  # Choose a working dir
DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache

mkdir -p ${DATA_DIR}
python ${TRANSFORMERS_PATH}/utils/download_glue_data.py \
    --data_dir ${DATA_DIR}/raw_glue_data

python jiant/scripts/preproc/export_glue_data.py \
    --input_base_path ${DATA_DIR}/raw_glue_data \
    --output_base_path ${DATA_DIR}/glue

python jiant/scripts/preproc/export_model.py \
    --model_type roberta-base \
    --output_base_path ${MODELS_DIR}/roberta-base

python jiant/proj/simple/tokenize_and_cache.py \
    --task_config_path ${DATA_DIR}/glue/configs/cola.json \
    --model_type roberta-large \
    --model_tokenizer_path ${MODELS_DIR}/roberta-base/tokenizer \
    --phases train,val,test \
    --max_seq_length 256 \
    --do_iter \
    --force_overwrite \
    --smart_truncate \
    --output_dir ${WORKING_DIR}/cache/cola

ls ${WORKING_DIR}/cache/cola
```