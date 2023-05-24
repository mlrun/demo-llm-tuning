import json
import os
import zipfile

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

import torch
import shutil
import tempfile

from abc import ABC
from typing import Any, Dict, List

import mlrun
import numpy as np
import pandas as pd
import transformers

from datasets import Dataset
from mlrun import MLClientCtx

from mlrun.artifacts import Artifact, PlotlyArtifact
from mlrun.datastore import DataItem
from mlrun.frameworks._common import CommonTypes, MLRunInterface
from mlrun.utils import create_class
from plotly import graph_objects as go

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

DEEPSPEED_CONFIG = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
    "comms_logger": {
      "enabled": True,
      "verbose": False,
      "prof_all": True,
      "debug": False
    }
}

# ----------------------from MLRUN--------------------------------
class HFTrainerMLRunInterface(MLRunInterface, ABC):
    """
    Interface for adding MLRun features for tensorflow keras API.
    """

    # MLRuns context default name:
    DEFAULT_CONTEXT_NAME = "mlrun-huggingface"

    # Attributes to replace so the MLRun interface will be fully enabled.
    _REPLACED_METHODS = [
        "train",
        # "evaluate"
    ]

    @classmethod
    def add_interface(
        cls,
        obj: Trainer,
        restoration: CommonTypes.MLRunInterfaceRestorationType = None,
    ):
        super(HFTrainerMLRunInterface, cls).add_interface(
            obj=obj, restoration=restoration
        )

    @classmethod
    def mlrun_train(cls):
        def wrapper(self: Trainer, *args, **kwargs):
            # Restore the evaluation method as `train` will use it:
            # cls._restore_attribute(obj=self, attribute_name="evaluate")

            # Call the original fit method:
            result = self.original_train(*args, **kwargs)

            # Replace the evaluation method again:
            # cls._replace_function(obj=self, function_name="evaluate")

            return result

        return wrapper


class MLRunCallback(TrainerCallback):
    """
    Callback for collecting logs during training / evaluation of the `Trainer` API.
    """

    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        model_name: str = "model",
        tag: str = "",
        labels: Dict[str, str] = None,
        extra_data: dict = None,
    ):
        super().__init__()

        # Store the configurations:
        self._context = (
            context
            if context is not None
            else mlrun.get_or_create_ctx("./mlrun-huggingface")
        )
        self._model_name = model_name
        self._tag = tag
        self._labels = labels
        self._extra_data = extra_data if extra_data is not None else {}

        # Set up the logging mode:
        self._is_training = False
        self._steps: List[List[int]] = []
        self._metric_scores: Dict[str, List[float]] = {}
        self._artifacts: Dict[str, Artifact] = {}

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._steps.append([])

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        recent_logs = state.log_history[-1].copy()

        recent_logs.pop("epoch")
        current_step = int(recent_logs.pop("step"))
        if current_step not in self._steps[-1]:
            self._steps[-1].append(current_step)

        for metric_name, metric_score in recent_logs.items():
            if metric_name.startswith("train_"):
                if metric_name.split("train_")[1] not in self._metric_scores:
                    self._metric_scores[metric_name] = [metric_score]
                continue
            if metric_name not in self._metric_scores:
                self._metric_scores[metric_name] = []
            self._metric_scores[metric_name].append(metric_score)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._is_training = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        self._log_metrics()

        if self._is_training:
            return

    def _log_metrics(self):
        for metric_name, metric_scores in self._metric_scores.items():
            self._context.log_result(key=metric_name, value=metric_scores[-1])
            if len(metric_scores) > 1:
                self._log_metric_plot(name=metric_name, scores=metric_scores)
        self._context.commit(completed=False)

    def _log_metric_plot(self, name: str, scores: List[float]):
        # Initialize a plotly figure:
        metric_figure = go.Figure()

        # Add titles:
        metric_figure.update_layout(
            title=name.capitalize().replace("_", " "),
            xaxis_title="Samples",
            yaxis_title="Scores",
        )

        # Draw:
        metric_figure.add_trace(
            go.Scatter(x=np.arange(len(scores)), y=scores, mode="lines")
        )

        # Create the plotly artifact:
        artifact_name = f"{name}_plot"
        artifact = PlotlyArtifact(key=artifact_name, figure=metric_figure)
        self._artifacts[artifact_name] = self._context.log_artifact(artifact)


def apply_mlrun(
    trainer: transformers.Trainer,
    model_name: str = None,
    tag: str = "",
    context: mlrun.MLClientCtx = None,
    auto_log: bool = True,
    labels: Dict[str, str] = None,
    extra_data: dict = None,
    **kwargs,
):
    # Get parameters defaults:
    if context is None:
        context = mlrun.get_or_create_ctx(HFTrainerMLRunInterface.DEFAULT_CONTEXT_NAME)

    HFTrainerMLRunInterface.add_interface(obj=trainer)

    if auto_log:
        trainer.add_callback(
            MLRunCallback(
                context=context,
                model_name=model_name,
                tag=tag,
                labels=labels,
                extra_data=extra_data,
            )
        )


class KWArgsPrefixes:
    MODEL_CLASS = "CLASS_"
    FIT = "FIT_"
    TRAIN = "TRAIN_"
    PREDICT = "PREDICT_"
    DATA_COLLATOR = "DC_"


def _get_sub_dict_by_prefix(src: Dict, prefix_key: str) -> Dict[str, Any]:
    return {
        key.replace(prefix_key, ""): val
        for key, val in src.items()
        if key.startswith(prefix_key)
    }


# ---------------------- Hugging Face Trainer --------------------------------
def train(
    context: MLClientCtx,
    dataset: DataItem = None,
    pretrained_tokenizer: str = None,
    pretrained_model: str = None,
    model_class: str = None,
    tokenizer_class: str = None,
    model_name: str = "huggingface-model",
    use_deepspeed: bool = True,
):
    torch.cuda.empty_cache()
    deepspeed_config_json = None
    if use_deepspeed:
        deepspeed_config_json = os.path.join(tempfile.mkdtemp(), "ds_config.json")
        with open(deepspeed_config_json, "w") as f:
            json.dump(DEEPSPEED_CONFIG, f)
    # Creating tokenizer:
    if tokenizer_class:
        tokenizer_class = create_class(tokenizer_class)
    else:
        tokenizer_class = AutoTokenizer

    tokenizer = tokenizer_class.from_pretrained(
        pretrained_tokenizer, model_max_length=512
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_pandas(dataset.as_df())

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = None

    data_collator_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.DATA_COLLATOR
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, **data_collator_kwargs
    )

    # Parsing kwargs:
    train_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.TRAIN
    )
    if use_deepspeed:
        train_kwargs["deepspeed"] = deepspeed_config_json
    model_class_kwargs = _get_sub_dict_by_prefix(
        src=context.parameters, prefix_key=KWArgsPrefixes.MODEL_CLASS
    )

    # Loading our pretrained model:
    model_class_kwargs["pretrained_model_name_or_path"] = (
        model_class_kwargs.get("pretrained_model_name_or_path") or pretrained_model
    )
    train_kwargs["hub_token"] = train_kwargs.get("hub_token") or pretrained_tokenizer
    if not model_class_kwargs["pretrained_model_name_or_path"]:
        raise mlrun.errors.MLRunRuntimeError(
            "Must provide pretrained_model name as "
            "function argument or in extra params"
        )
    model = create_class(model_class).from_pretrained(**model_class_kwargs)

    # Preparing training arguments:
    training_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        **train_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    apply_mlrun(trainer, model_name=model_name)

    # Apply training with evaluation:
    context.logger.info(f"training '{model_name}'")
    trainer.train()

    temp_directory = tempfile.gettempdir()
    trainer.save_model(temp_directory)

    # Zip the model directory:
    shutil.make_archive(
        base_name="model",
        format="zip",
        root_dir=temp_directory,
    )

    # Log the model:
    context.log_model(
        key="model",
        db_key=model_name,
        model_file="model.zip",
        tag="",
        framework="Hugging Face",
    )


def evaluate(context, model_path, data: pd.DataFrame):
    # Get the model artifact and file:
    (
        model_file,
        model_artifact,
        extra_data,
    ) = mlrun.artifacts.get_model(model_path)

    # Read the name:
    model_name = model_artifact.spec.db_key

    # Extract logged model files:
    model_directory = os.path.join(os.path.dirname(model_file), model_name)
    with zipfile.ZipFile(model_file, "r") as zip_file:
        zip_file.extractall(model_directory)

    # Loading the saved pretrained tokenizer and model:
    dataset = Dataset.from_pandas(data)
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    encodings = tokenizer("\n\n".join(dataset["text"][:5]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    context.log_result("perplexity", ppl)
