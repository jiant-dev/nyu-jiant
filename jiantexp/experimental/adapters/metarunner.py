import os

import torch
import torch.nn as nn

import jiant.proj.main.metarunner as jiant_metarunner
import jiantexp.experimental.adapters.modeling as adapters_modeling
import jiant.utils.python.io as py_io
import jiant.utils.torch_utils as torch_utils

import jiant.proj.main.components.task_sampler as jiant_task_sampler
from jiant.utils.torch_utils import copy_state_dict, CPU_DEVICE, get_model_for_saving


def save_model_with_metadata(model: nn.Module, metadata: dict, output_dir: str, file_name="model"):
    torch.save(
        adapters_modeling.get_optimized_state_dict_for_jiant_model_with_adapters(
            torch_utils.get_model_for_saving(model)
        ),
        os.path.join(output_dir, f"{file_name}.p"),
    )
    py_io.write_json(metadata, os.path.join(output_dir, f"{file_name}.metadata.json"))


class AdaptersMetarunner(jiant_metarunner.JiantMetarunner):

    # This metarunner modifies the original metarunner to support adapter workflows.
    # Specifically, we want to only save the tuned-parameters to best_model.p
    # We, however, do not modify the checkpoint-saving or best_state_dict
    # because the current metarunner API doesn't make it easy to modify that

    def save_model(self):
        """Override to save only optimized parameters"""
        save_model_with_metadata(
            model=self.model,
            metadata={},
            output_dir=self.output_dir,
            file_name=f"model__{self.train_state.global_steps:09d}",
        )

    def eval_save(self):
        self.num_evals_since_improvement += 1
        val_results_dict = self.runner.run_val(
            task_name_list=self.runner.jiant_task_container.task_run_config.train_val_task_list,
            use_subset=True,
        )
        aggregated_major = jiant_task_sampler.compute_aggregate_major_metrics_from_results_dict(
            metrics_aggregator=self.runner.jiant_task_container.metrics_aggregator,
            results_dict=val_results_dict,
        )
        val_metrics_dict = jiant_task_sampler.get_metrics_dict_from_results_dict(
            results_dict=val_results_dict,
        )
        val_state = jiant_metarunner.ValState(
            score=float(aggregated_major),
            metrics=val_metrics_dict,
            train_state=self.train_state.new(),
        )
        self.log_writer.write_entry("train_val", val_state.to_dict())
        if self.best_val_state is None or val_state.score > self.best_val_state.score:
            self.best_val_state = val_state.new()
            self.log_writer.write_entry("train_val_best", self.best_val_state.to_dict())
            if self.save_best_model:
                save_model_with_metadata(
                    model=self.model,
                    metadata={
                        "val_state": self.best_val_state.to_dict(),
                        "val_metrics": val_metrics_dict,
                    },
                    output_dir=self.output_dir,
                    file_name="best_model",
                )
            self.best_state_dict = copy_state_dict(
                state_dict=get_model_for_saving(self.model).state_dict(), target_device=CPU_DEVICE,
            )
            self.num_evals_since_improvement = 0
        self.log_writer.write_entry(
            "early_stopping",
            {
                "num_evals_since_improvement": self.num_evals_since_improvement,
                "train_state": self.train_state.to_dict(),
            },
        )
        self.log_writer.flush()
        self.val_state_history.append(val_state)
