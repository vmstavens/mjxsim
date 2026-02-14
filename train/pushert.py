import json
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset

from agents.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    ConditionalUnet1D,
    DiffusionPolicy,
    EMAModel,
)
from datasets.pushert import PushTStateDataset, download_dataset
from trainers.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
)
from utils import demo as loc_demo
from utils.datasets import split_dataset

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)


epochs, avg_losses, val_losses = [], [], []
_HERE = Path(__file__).parent.parent
_SAVE_PATH = _HERE / ".demo/state/"
_SAVE_PATH.mkdir(parents=True, exist_ok=True)
_SAVE_PATH_MODELS = _HERE / ".demo/state/models/"
_SAVE_PATH_MODELS.mkdir(parents=True, exist_ok=True)
_SAVE_PATH_MEDIA = _HERE / ".demo/state/media/"
_SAVE_PATH_MEDIA.mkdir(parents=True, exist_ok=True)
dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG
trainer_config = SUPERVISED_TRAINER_DEFAULT_CONFIG

dp_config["eval_frequency"] = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_path = download_dataset()

dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=dp_config["pred_horizon"],
    obs_horizon=dp_config["obs_horizon"],
    action_horizon=dp_config["action_horizon"],
)


# train_dataset, valid_dataset = split_dataset(dataset=dataset)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=dp_config["batch_size"],
    num_workers=dp_config["num_workers"],
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True,
)


a_dim = 2
o_dim = 5

# Build models
dp_models = {}
dp_models["model"] = ConditionalUnet1D(a_dim, dp_config).to(device)
ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
dp_models["ema_model"] = ConditionalUnet1D(a_dim, dp_config).to(device)

# Create agent (assuming you have appropriate models)
# You'll need to provide actual model instances here
agent = DiffusionPolicy(
    models=dp_models, ema=ema, config=dp_config, dataloader=dataloader
)


def _cb(epoch: int, avg_loss: int) -> None:
    epochs.append(epoch)
    avg_losses.append(avg_loss)
    # val_losses.append(val_loss)

    # Save training plot
    plt.plot(epochs, avg_losses, label="train loss")
    # plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig((_SAVE_PATH / "training_plot.png").as_posix())
    plt.close()

    # Save model directly to the models directory with epoch number
    model_path = _SAVE_PATH_MODELS / f"model_epoch_{epoch}.pth"
    agent.save(model_path.as_posix())

    # Also save as latest model
    latest_path = _SAVE_PATH_MODELS / "latest_model.pth"
    agent.save(latest_path.as_posix())

    agent.is_trained = True
    agent.ema.copy_to(agent.ema_model.parameters())
    max_reward, frames = loc_demo.rollout(
        agent, dataset.stats, max_steps=dp_config["max_steps"]
    )
    video_path = _SAVE_PATH_MEDIA / f"epoch_{epoch:02d}_reward_{max_reward:04f}.mp4"
    demo.save_video(frames, video_path.as_posix(), verbose=False)
    agent.is_trained = False


agent.set_mode("train")

# Create supervised trainer
trainer = SupervisedTrainer(
    agent=agent,
    trainer_config=dp_config,
    # trainer_config=trainer_config,
    train_loader=dataloader,
    # valid_loader=valid_loader,
    callback_fn=_cb,
)

# Train the model
trainer.train()

# # Evaluate the model
# trainer.eval()
