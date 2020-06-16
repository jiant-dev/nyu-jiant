import os
import shutil

import jiant.utils.zconf as zconf
import jiant.scripts.preproc.export_glue_data as export_glue_data

PHILS_ARCHIVE_BASE_PATH = "/archive/p/pcy214/public/task_data"


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    task_name = zconf.attr(type=str, required=True)
    data_dir = zconf.attr(type=str, required=True)


def simply_copy_fol(data_dir, original_fol_name, task_name):
    shutil.copytree(
        src=os.path.join(PHILS_ARCHIVE_BASE_PATH, original_fol_name),
        dst=os.path.join(data_dir, task_name),
    )


def copy_glue_data(data_dir, task_name):
    export_glue_data.convert_glue_data(
        input_base_path=PHILS_ARCHIVE_BASE_PATH,
        task_data_path=os.path.join(data_dir, task_name),
        task_name=task_name,
    )


def fetch_data(task_name: str, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    if task_name == "boolq":
        simply_copy_fol(data_dir=data_dir, original_fol_name="BoolQ", task_name=task_name)
    elif task_name == "ccg":
        task_data_dir = os.path.join(data_dir, "ccg")
        os.makedirs(task_data_dir, exist_ok=True)
        for phase in ["train", "dev", "test"]:
            shutil.copy(
                src=os.path.join(PHILS_ARCHIVE_BASE_PATH, "CCG", f"ccg.{phase}"),
                dst=os.path.join(task_data_dir, f"ccg.{phase}"),
            )
    elif task_name == "cola":
        copy_glue_data(data_dir=data_dir, task_name=task_name)
    elif task_name == "hellaswag":
        simply_copy_fol(data_dir=data_dir, original_fol_name="HellaSwag", task_name=task_name)
    elif task_name == "mnli":
        copy_glue_data(data_dir=data_dir, task_name=task_name)
    elif task_name == "rte":
        copy_glue_data(data_dir=data_dir, task_name=task_name)
    elif task_name == "squad_v1":
        simply_copy_fol(data_dir=data_dir, original_fol_name="SQuADv1", task_name=task_name)
    elif task_name == "wic":
        simply_copy_fol(data_dir=data_dir, original_fol_name="WiC", task_name=task_name)
    else:
        raise KeyError(task_name)


if __name__ == "__main__":
    args = RunConfiguration.default_run_cli()
    fetch_data(
        task_name=args.task_name,
        data_dir=args.data_dir,
    )
