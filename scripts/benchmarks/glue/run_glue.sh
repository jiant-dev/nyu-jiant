DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache
RUN_CONFIG_DIR=${WORKING_DIR}/run_configs/glue
OUTPUT_DIR=${WORKING_DIR}/output_dir/glue

GLUE_TASKS=(cola mrpc qqp sst stsb mnli mnli_mm qnli rte glue_diagnostics)
MODEL_TYPE=roberta-large

# Generate run configs
for TASK_NAME in "${GLUE_TASKS[@]}"
do
   python glue_run_config.py $TASK_NAME
done

# Run training
for TASK_NAME in "${GLUE_TASKS[@]}"
do
	echo $TASK_NAME
    sbatch --export=DATA_DIR=$DATA_DIR,MODELS_DIR=$MODELS_DIR,CACHE_DIR=$CACHE_DIR,RUN_CONFIG_DIR=$RUN_CONFIG_DIR,OUTPUT_DIR=$OUTPUT_DIR,TASK_NAME=$TASK_NAME,MODEL_TYPE=$MODEL_TYPE glue_task.sbatch
done
