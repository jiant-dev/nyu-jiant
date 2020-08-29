DATA_DIR=${WORKING_DIR}/data
MODELS_DIR=${WORKING_DIR}/models
CACHE_DIR=${WORKING_DIR}/cache

SUPERGLUE_TASKS=(cb copa multirc wic wsc boolq record superglue_broadcoverage_diagnostics superglue_winogender_diagnostics)
MODEL_TYPE=roberta-large

python ${JIANT_PATH}/scripts/download_data/runscript.py download --benchmark SUPERGLUE --output_path $DATA_DIR

python ${JIANT_PATH}/proj/main/export_model.py \
   --model_type ${MODEL_TYPE} \
   --output_base_path ${MODELS_DIR}/${MODEL_TYPE}

for TASK_NAME in "${SUPERGLUE_TASKS[@]}"
do
    if [ "$TASK_NAME" == "superglue_broadcoverage_diagnostics" ] || [ "$TASK_NAME" == "superglue_winogender_diagnostics" ]; then
        phases=test
    else
        phases=train,val,test
    fi
    
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
