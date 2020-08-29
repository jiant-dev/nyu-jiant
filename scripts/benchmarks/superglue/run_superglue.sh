DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_configs/superglue
OUTPUT_DIR=${WORKING_DIR}/output_dir/superglue

SUPERGLUE_TASKS=(cb copa multirc wic wsc boolq record superglue_broadcoverage_diagnostics superglue_winogender_diagnostics)
MODEL_TYPE=roberta-large

# Generate run configs
for TASK_NAME in "${SUPERGLUE_TASKS[@]}"
do
   python superglue_run_configs.py $TASK_NAME $MODEL_TYPE
done

# Run training
for TASK_NAME in "${SUPERGLUE_TASKS[@]}"
do
	echo $TASK_NAME
    sbatch --export=DATA_DIR=$DATA_DIR,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=$RUN_CONFIG_DIR,OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE glue_task.sbatch
done
