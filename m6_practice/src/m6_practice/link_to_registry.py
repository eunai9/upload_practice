import wandb
api = wandb.Api()
artifact_path = "eunai9/corrupt_mnist/corrupt_mnist_model:latest"
artifact = api.artifact(artifact_path)
artifact.link(target_path="eunai9/model-registry/corrupt-mnist-model")
artifact.save()