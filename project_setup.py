import importlib

import mlrun


def assert_build():
    for module_name in [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "evaluate",
        "deepspeed",
        "mpi4py",
    ]:
        module = importlib.import_module(module_name)
        print(module.__version__)


def setup(
        project: mlrun.projects.MlrunProject
):
    """
    Creating the project for this demo.
    :returns: a fully prepared project for this demo.
    """
    print(project.get_param("source"))
    # Set or build the default image:
    if project.get_param("default_image") is None:
        print("Building image for the demo:")
        image_builder = project.set_function(
            "project_setup.py",
            name="image-builder",
            handler="assert_build",
            kind="job",
            image="mlrun/mlrun-gpu",
            requirements=[
                "torch",
                "transformers[deepspeed]",
                "datasets",
                "accelerate",
                "evaluate",
                "mpi4py",
            ],
        )
        assert image_builder.deploy()
        default_image = image_builder.spec.image
    project.set_default_image(project.get_param("default_image"))

    # Set the project git source:

    project.set_source(project.get_param("source"), pull_at_runtime=True)

    # Set the data collection function:
    data_collection_function = project.set_function(
        "src/data_collection.py",
        name="data-collecting",
        image="mlrun/mlrun",
        kind="job",

    )
    data_collection_function.apply(mlrun.auto_mount())
    data_collection_function.save()

    # Set the data preprocessing function:
    project.set_function(
        "src/data_preprocess.py",
        name="data-preparing",
        kind="job",
    )

    # Set the training function:
    train_function = project.set_function(
        "src/trainer.py",
        name="training",
        kind="job",
    )
    train_function.with_limits(
        gpus=project.get_param("num_gpus_per_replica") or 4,
        cpu=project.get_param("num_cpus_per_replica") or 48,
        mem=project.get_param("memory_per_replica") or "192Gi",
    )
    train_function.save()

    project.set_function(
        "src/serving.py",
        name="serving",
        kind="serving",
    )

    # Set the training workflow:
    project.set_workflow("training_workflow", "src/training_workflow.py")

    # Save and return the project:
    project.save()
    return project