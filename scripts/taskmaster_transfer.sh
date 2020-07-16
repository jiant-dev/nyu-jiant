DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/taskmaster_transfer
OUTPUT_DIR=${WORKING_DIR}/output_dir/taskmaster_transfer

TARGET_TASKS=(boolq copa)
INTERMEDIATE_TASKS=(commonsenseqa)
TASKS=( "${TARGET_TASKS[@]}" "${INTERMEDIATE_TASKS[@]}" )

MODEL_TYPE=roberta-large

# defaults from https://github.com/nyu-mll/jiant/blob/taskmaster_v1/jiant/config/taskmaster/base_roberta.conf
epochs = 10
val_interval = 5000
lr = 0.00001

# Generate run configs
for TASK_NAME in "${TASKS[@]}"
do
    echo ${TASK_NAME}

    # target tasks
    if [ "${TASK_NAME}" == "boolq" ]; then
        val_interval = 2400
        train_batch_size=4
        lr=0.000005 
    elif [ "${TASK_NAME}" == "cb" ]; then
        val_interval = 60
        epochs = 40
        train_batch_size=4
        lr=0.00005
    elif [ "${TASK_NAME}" == "commonsenseqa" ]; then
        val_interval = 2500
        train_batch_size=4
        lr=0.000003
    elif [ "${TASK_NAME}" == "copa" ]; then
        val_interval = 100
        epochs = 40
        train_batch_size=32
        lr=0.000005
    elif [ "${TASK_NAME}" == "cosmosqa" ]; then
        train_batch_size=4
        lr=0.000003
    elif [ "${TASK_NAME}" == "mrc" ]; then
        val_interval = 1000
        train_batch_size=4
       	lr=0.00002
    elif [ "${TASK_NAME}" == "record" ]; then
       	train_batch_size=4
        lr=0.00005
    elif [ "${TASK_NAME}" == "rte" ]; then
        val_interval = 625
        train_batch_size=4
        lr=0.000005
    elif [ "${TASK_NAME}" == "wic" ]; then
        val_interval = 1000
        train_batch_size=32
        lr=0.00005
    #intermediate tasks (commonsenseqa and cosmosqa handled above)
    elif [ "${TASK_NAME}" == "sst" ]; then
        train_batch_size=64
        lr=0.000003
    elif [ "${TASK_NAME}" == "socialiqa" ]; then
        train_batch_size=4
        lr=0.00002
    elif [ "${TASK_NAME}" == "qqp" ]; then
        train_batch_size=8
        lr=0.000005
    elif [ "${TASK_NAME}" == "mnli" ]; then
        train_batch_size=4
        lr=0.000003
    elif [ "${TASK_NAME}" == "scitail" ]; then
        train_batch_size=4
        lr=0.000005
    elif [ "${TASK_NAME}" == "squad" ]; then
        train_batch_size=4
        lr=0.000005
    elif [ "${TASK_NAME}" == "hellaswag" ]; then
        train_batch_size=4
        lr=0.000003
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
    sbatch --export=DATA_DIR=$DATA_DIR,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=$RUN_CONFIG_DIR,OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE,VAL_INTERVAL=$val_interval,TARGET_TASKS=$TARGET_TASKS,LR=$lr intermediate_target_task.sbatch
done
