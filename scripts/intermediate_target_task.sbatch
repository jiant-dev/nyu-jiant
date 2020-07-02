#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:p40:1
#SBATCH --time=72:00:00
#SBATCH --mem=40GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=js11133@nyu.edu
#SBATCH --output=/scratch/js11133/jiant_working_dir/slurm_output/slurm_%j.out

module purge
module load python3/intel/3.7.3
source activate /home/js11133/.conda/envs/jiant

echo $MODELS_DIR
echo $MODEL_TYPE
echo $RUN_CONFIG_DIR

timestamp=$(date +%s)

#intermediate task
python \
    $JIANT_PATH/proj/main/runscript.py   \
    run \
    --ZZsrc ${MODELS_DIR}/${MODEL_TYPE}/config.json \
    --jiant_task_container_config_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json \
    --model_load_mode from_transformers \
    --learning_rate 1e-5 \
    --force_overwrite \
    --do_train \
    --do_val \
    --do_save \
    --eval_every_steps $VAL_INTERVAL \
    --no_improvements_for_n_evals 30 \
    --save_checkpoint_every_steps 10000 \
    --output_dir ${OUTPUT_DIR}/${TASK_NAME}/ \
    --seed $timestamp


for TARGET_TASK in "${TARGET_TASK[@]}"
do
    python \
	    $JIANT_PATH/proj/main/runscript.py   \
	    run \
            --ZZoverrides model_path
   	    --ZZsrc ${MODELS_DIR}/${MODEL_TYPE}/config.json \
   	    --jiant_task_container_config_path ${RUN_CONFIG_DIR}/${TASK_NAME}.json \
   	    --model_load_mode from_partial \
            --model_path ${OUTPUT_DIR}/${TASK_NAME}/best_model.p
   	    --learning_rate 1e-5 \
   	    --force_overwrite \
   	    --do_train \
   	    --do_val \
   	    --do_save \
   	    --eval_every_steps $VAL_INTERVAL \
   	    --no_improvements_for_n_evals 30 \
   	    --save_checkpoint_every_steps 10000 \
   	    --output_dir ${OUTPUT_DIR}/${TASK_NAME}/${TARGET_TASK} \
	    --seed $timestamp
done
