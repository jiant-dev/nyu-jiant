DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/taskmaster_transfer
OUTPUT_DIR=${WORKING_DIR}/output_dir/taskmaster_transfer

# Not supported: qa-srl, qamr
# Error in wsc (#78)
# TASKMASTER_TASKS=(boolq ccg cb commonsenseqa copa cosmosqa hellaswag mnli mrc qqp record rte scitail socialiqa sst wic wsc)
TARGET_TASKS=(boolq cb commonsenseqa copa rte)
INTERMEDIATE_TASKS=(mnli qqp)
TASKS=( "${TARGET_TASKS[@]}" "${INTERMEDIATE_TASKS[@]}" )

MODEL_TYPE=roberta-large

# runscript default arguments
# from: https://github.com/nyu-mll/jiant/blob/taskmaster_v1/scripts/taskmaster_v1/all_tasks.sh
train_batch_size=4
val_interval=1000
epochs=10

# Generate run configs
for TASK_NAME in "${TASKS[@]}"
do
    echo ${TASK_NAME}

    if [ "${TASK_NAME}" == "boolq" ]; then
        val_interval=1000
    elif [ "${TASK_NAME}" == "cb" ]; then
       	val_interval=60
        epochs=40
    elif [ "${TASK_NAME}" == "copa" ]; then
        train_batch_size=100
        epochs=40
        learning_rate=2e-5
    elif [ "${TASK_NAME}" == "cosmosqa" ]; then
        train_batch_size=3 #debug param
    elif [ "${TASK_NAME}" == "multirc" ]; then
        train_batch_size=1 #debug param
       	val_interval=1000
    elif [ "${TASK_NAME}" == "record" ]; then
       	train_batch_size=8
        val_interval=10000
    elif [ "${TASK_NAME}" == "rte" ]; then
        val_interval=625
    elif [ "${TASK_NAME}" == "wic" ]; then
        val_interval=1000  
    fi

    python ${NYU_JIANT_DIR}/documentation/porting_examples/example4_assets/make_config.py \
        --task_config_path ${DATA_DIR}/configs/${TASK_NAME}.json \
        --task_cache_base_path ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME} \
        --train_batch_size $train_batch_size \
        --epochs $epochs \
        --output_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json
done

# Run training
for TASK_NAME in "${INTERMEDIATE_TASKS[@]}"
do  
    echo $TASK_NAME
    sbatch --export=DATA_DIR=$DATA_DIR,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=$RUN_CONFIG_DIR,OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE,VAL_INTERVAL=$val_interval,TARGET_TASKS=$TARGET_TASKS intermediate_target_task.sbatch
done
