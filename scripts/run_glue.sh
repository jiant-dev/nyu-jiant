DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/metarunner_example
OUTPUT_DIR=${WORKING_DIR}/output_dir/metarunner_example

# ERROR with wnli, diagnostic, qnli in setup
GLUE_TASKS=(cola mrpc qnli qqp sst stsb mnli rte)
MODEL_TYPE=bert-base-cased

# Generate run configs
for TASK_NAME in "${GLUE_TASKS[@]}"
do
    python ${NYU_JIANT_DIR}/documentation/porting_examples/example4_assets/make_config.py \
        --task_config_path ${DATA_DIR}/glue/configs/${TASK_NAME}.json \
        --task_cache_base_path ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME} \
        --train_batch_size 32 \
        --epochs 3 \
        --output_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json
    echo ${TASK_NAME}
done

# Run training
for TASK_NAME in "${GLUE_TASKS[@]}"
do
    sbatch --export=DATA_DIR=$DATA_DIR,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=$RUN_CONFIG_DIR,OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE glue_task.sbatch
    echo $TASK_NAME
done
