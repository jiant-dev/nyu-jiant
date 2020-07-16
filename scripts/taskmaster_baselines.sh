DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_config_dir/taskmaster
OUTPUT_DIR=${WORKING_DIR}/output_dir/taskmaster

# Not supported: qa-srl, qamr
# Error in wsc (#78)
# TASKMASTER_TASKS=(boolq ccg cb commonsenseqa copa cosmosqa hellaswag mnli mrc qqp record rte scitail socialiqa sst wic wsc)
TASKMASTER_TARGET_TASKS=${1:-(boolq cb ccg commonsenseqa copa cosmosqa hellaswag mnli mrc qqp record rte scitail socialiqa sst wic)}
MODEL_TYPE=roberta-large

# Generate run configs
for TASK_NAME in "${TASKMASTER_TARGET_TASKS[@]}"
do
    echo ${TASK_NAME}

    # defaults from https://github.com/nyu-mll/jiant/blob/taskmaster_v1/jiant/config/taskmaster/base_roberta.conf
    train_batch_size=4
    val_interval=5000
    epochs=10
    lr=0.00001

    # target tasks
    if [ "${TASK_NAME}" == "boolq" ]; then
        val_interval=2400
        train_batch_size=4
        lr=0.000005 
    elif [ "${TASK_NAME}" == "cb" ]; then
        val_interval=60
        epochs=40
        train_batch_size=4
        lr=0.00005
    elif [ "${TASK_NAME}" == "commonsenseqa" ]; then
        val_interval=2500
        train_batch_size=4
        lr=0.000003
    elif [ "${TASK_NAME}" == "copa" ]; then
        val_interval=100
        epochs=40
        train_batch_size=32
        lr=0.000005
    elif [ "${TASK_NAME}" == "cosmosqa" ]; then
        train_batch_size=4
        lr=0.000003
    elif [ "${TASK_NAME}" == "mrc" ]; then
        val_interval=1000
        train_batch_size=4
       	lr=0.00002
    elif [ "${TASK_NAME}" == "record" ]; then
       	train_batch_size=4
        lr=0.00005
    elif [ "${TASK_NAME}" == "rte" ]; then
        val_interval=625
        train_batch_size=4
        lr=0.000005
    elif [ "${TASK_NAME}" == "wic" ]; then
        val_interval=1000
        train_batch_size=32
        lr=0.00005
    fi

    python ${NYU_JIANT_DIR}/documentation/porting_examples/example4_assets/make_config.py \
        --task_config_path ${DATA_DIR}/configs/${TASK_NAME}.json \
        --task_cache_base_path ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME} \
        --train_batch_size $train_batch_size \
        --epochs $epochs \
        --output_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json

    sbatch --export=DATA_DIR=$DATA_DIR,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=$RUN_CONFIG_DIR,OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE,VAL_INTERVAL=$val_interval,LR=$lr task.sbatch
done
