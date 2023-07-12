import mlrun
from kfp import dsl


@dsl.pipeline(name="MLOps Bot Master Pipeline")
def kfpipeline(
    html_links: str,
    model_name: str,
    pretrained_tokenizer: str,
    pretrained_model: str,
    epochs: str,
    use_deepspeed: bool,
    tokenizer_class: str = "transformers.AutoTokenizer",
    model_class: str = "transformers.AutoModelForCausalLM",
):
    # Get our project object:
    project = mlrun.get_current_project()

    # Collect Dataset:
    collect_dataset_run = mlrun.run_function(
        function="data-collecting",
        handler="collect_html_to_text_files",
        name="data-collection",
        params={"urls_file": html_links},
        returns=["html-as-text-files:path"],
    )

    # Dataset Preparation:
    prepare_dataset_run = mlrun.run_function(
        function="data-preparing",
        handler="prepare_dataset",
        name="data-preparation",
        inputs={"source_dir": collect_dataset_run.outputs["html-as-text-files"]},
        returns=["html-data:dataset"],
    )

    # Training:
    project.get_function("training")

    training_run = mlrun.run_function(
        function="training",
        name="train",
        inputs={"dataset": prepare_dataset_run.outputs["html-data"]},
        params={
            "model_name": model_name,
            "pretrained_tokenizer": pretrained_tokenizer,
            "pretrained_model": pretrained_model,
            "model_class": model_class,
            "tokenizer_class": tokenizer_class,
            "TRAIN_num_train_epochs": epochs,
            "use_deepspeed": use_deepspeed,
        },
        handler="train",
        outputs=["model"],
    )

    # evaluation:
    mlrun.run_function(
        function="training",
        name="evaluate",
        params={
            "model_path": training_run.outputs["model"],
            "model_name": pretrained_model,
            "tokenizer_name": pretrained_tokenizer,
        },
        inputs={"data": prepare_dataset_run.outputs["html-data"]},
        handler="evaluate",
    )
