import sys
sys.path.insert(0,"/home/js11133/jiant")
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.runscript as main_runscript
import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.display as display
import os
import torch
from pathlib import Path


task_name = sys.argv[1]
GLUE_RUN_CONFIG_DIR = "/scratch/js11133/jiant_working_dir/run_configs/glue/"
Path(GLUE_RUN_CONFIG_DIR).mkdir(parents=True, exist_ok=True)

if task_name == "glue_diagnostics" or task_name == "mnli_mismatched":
	jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
	    task_config_base_path="/scratch/js11133/jiant_working_dir/data/configs",
	    task_cache_base_path="/scratch/js11133/jiant_working_dir/cache/roberta-large",
	    train_task_name_list=["mnli"],
	    test_task_name_list=[task_name],
	    train_batch_size=16,
	    eval_batch_size=16,
	    num_gpus=1,
	    epochs=0.01,
	).create_config()
else:
	jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
	    task_config_base_path="/scratch/js11133/jiant_working_dir/data/configs",
	    task_cache_base_path="/scratch/js11133/jiant_working_dir/cache/roberta-large",
	    train_task_name_list=[task_name],
	    test_task_name_list=[task_name],
	    train_batch_size=16,
	    eval_batch_size=16,
	    num_gpus=1,
	    epochs=0.01,
	).create_config()
py_io.write_json(jiant_run_config, GLUE_RUN_CONFIG_DIR + task_name + "_jiant_run_config.json")