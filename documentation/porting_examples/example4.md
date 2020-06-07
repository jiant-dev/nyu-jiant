# Metarunner example

```bash
WORKING_DIR=...  # Choose a working dir
NYU_JIANT_DIR=...  # https://github.com/jiant-dev/nyu-jiant

MODELS_DIR=${WORKING_DIR}/models
MODEL_TYPE=bert-base-uncased
DATA_DIR=${WORKING_DIR}/data
CACHE_DIR=${WORKING_DIR}/cache2
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/metarunner_example
OUTPUT_DIR=${WORKING_DIR}/output_dir/metarunner_example

# Download bert-base-uncased: it has clearer transfer performance
python jiant/scripts/preproc/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${MODELS_DIR}/${MODEL_TYPE}

# Tokenize and cache for bert-base-uncased
for TASK_NAME in mnli rte
do
    python jiant/proj/simple/tokenize_and_cache.py \
        --task_config_path ${DATA_DIR}/glue/configs/${TASK_NAME}.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${MODELS_DIR}/${MODEL_TYPE}/tokenizer \
        --phases train,val \
        --max_seq_length 256 \
        --do_iter \
        --smart_truncate \
        --output_dir ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME}
done

# Generate run configs
for TASK_NAME in mnli rte
do
    python ${NYU_JIANT_DIR}/documentation/porting_examples/example4_assets/make_config.py \
        --task_config_path ${DATA_DIR}/glue/configs/${TASK_NAME}.json \
        --task_cache_base_path ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME} \
        --train_batch_size 32 \
        --epochs 3 \
        --output_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json
done

# Train MNLI
python \
    jiant/proj/main/runscript.py \
    run \
    --ZZsrc ${MODELS_DIR}/${MODEL_TYPE}/config.json \
    --jiant_task_container_config_path ${RUN_CONFIG_DIR}/mnli.json \
    --model_load_mode from_transformers \
    --learning_rate 1e-5 \
    --force_overwrite \
    --do_train --do_val \
    --do_save \
    --eval_every_steps 5000 \
    --no_improvements_for_n_evals 30 \
    --save_checkpoint_every_steps 10000 \
    --output_dir ${OUTPUT_DIR}/mnli/

# Train MNLI->RTE
python \
    jiant/proj/main/runscript.py \
    run \
    --ZZoverrides model_path \
    --ZZsrc ${MODELS_DIR}/${MODEL_TYPE}/config.json \
    --jiant_task_container_config_path ${RUN_CONFIG_DIR}/rte.json \
    --model_load_mode partial \
    --model_path ${OUTPUT_DIR}/mnli/best_model.p \
    --learning_rate 1e-5 \
    --force_overwrite \
    --do_train --do_val \
    --do_save \
    --eval_every_steps 5000 \
    --no_improvements_for_n_evals 30 \
    --save_checkpoint_every_steps 10000 \
    --output_dir ${OUTPUT_DIR}/rte/

grep major ${OUTPUT_DIR}/rte/val_metrics.json
```