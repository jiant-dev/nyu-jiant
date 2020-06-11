#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p40:1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=glues-1gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=js11133@nyu.edu
module purge
module load python3/intel/3.7.3
source activate /home/js11133/.conda/envs/jiant

echo $MODELS_DIR
echo $MODEL_TYPE
echo $RUN_CONFIG_DIR

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
   	--eval_every_steps 1000 \
   	--no_improvements_for_n_evals 30 \
   	--save_checkpoint_every_steps 10000 \
   	--output_dir ${OUTPUT_DIR}/${TASK_NAME}/
