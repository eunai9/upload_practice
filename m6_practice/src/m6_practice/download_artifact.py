import wandb
from m6_practice.model import MyAwesomeModel
import torch

run = wandb.init()
artifact = run.use_artifact('eunai9/model-registry/corrupt-mnist-model:latest', type='model')
artifact_dir = artifact.download("wandb")
model = MyAwesomeModel()
model.load_state_dict(torch.load("wandb/model.pth"))