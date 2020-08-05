DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache

GLUE_TASKS=(cola mrpc qqp sst stsb mnli mnli_mm qnli rte wnli glue_diagnostics)
MODEL_TYPE=roberta-large


#python ${JIANT_PATH}/scripts/download_data/runscript.py download --benchmark GLUE --output_path $DATA_DIR

python ${JIANT_PATH}/proj/main/export_model.py \
   --model_type ${MODEL_TYPE} \
   --output_base_path ${MODELS_DIR}/${MODEL_TYPE}

for TASK_NAME in "${GLUE_TASKS[@]}"
do
	if [ "${TASK_NAME}" == "glue_diagnostics" ]; then
		phases=test
    elif [ "${TASK_NAME}" == "mnli_mm" ]
        phases=val,test
	else
		phases=train,val,test
	fi
	echo $phases
	python ${JIANT_PATH}/proj/main/tokenize_and_cache.py \
    		--task_config_path ${DATA_DIR}/configs/${TASK_NAME}_config.json \
    		--model_type ${MODEL_TYPE} \
    		--model_tokenizer_path ${MODELS_DIR}/${MODEL_TYPE}/tokenizer \
    		--phases ${phases} \
    		--max_seq_length 256 \
    		--do_iter \
    		--smart_truncate \
    		--output_dir ${CACHE_DIR}/${MODEL_TYPE}/${TASK_NAME}
done
