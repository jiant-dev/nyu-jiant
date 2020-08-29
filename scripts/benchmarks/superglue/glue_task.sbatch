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

SEED=42

python \
    $JIANT_PATH/proj/main/runscript.py   \
    run \
       --ZZsrc ${MODELS_DIR}/${MODEL_TYPE}/config.json \
       --jiant_task_container_config_path ${RUN_CONFIG_DIR}/${TASK_NAME}_jiant_run_config.json \
       --model_load_mode from_transformers \
       --learning_rate 0.00001 \
       --force_overwrite \
       --do_train \
       --do_val \
       --do_save \
       --eval_every_steps 500 \
       --no_improvements_for_n_evals 30 \
       --save_checkpoint_every_steps 10000 \
       --output_dir ${OUTPUT_DIR}/${TASK_NAME}/ \
       --seed $SEED \
       --write_test_preds 

