import logging
import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt

import agents.diffusion_policy_state as dp
from agents.diffusion_policy_state import DiffusionPolicy
from datasets.attractor import AttractorTrajectoryDataset
from trainers.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
)

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)

_HERE = Path(__file__).parent.parent
_SAVE_PATH = _HERE / f".demo/{__name__}/state/"
_SAVE_PATH.mkdir(parents=True, exist_ok=True)
_SAVE_PATH_MODELS = _HERE / ".demo/state/models/"
_SAVE_PATH_MODELS.mkdir(parents=True, exist_ok=True)
_SAVE_PATH_MEDIA = _HERE / ".demo/state/media/"
_SAVE_PATH_MEDIA.mkdir(parents=True, exist_ok=True)
dp_config = dp.DIFFUSION_POLICY_STATE_DEFAULT_CONFIG
trainer_config = SUPERVISED_TRAINER_DEFAULT_CONFIG
model_path = Path(__file__).parent.parent / ".runs"
model_path.mkdir(parents=True, exist_ok=True)

dp_config["experiment"]["directory"] = model_path.as_posix()
dp_config["experiment"]["experiment_name"] = Path(__file__).stem
dp_config["experiment"]["wandb"] = True
# trainer_config["write_interval"] = 1
# trainer_config["checkpoint_interval"] = 10
# trainer_config["batch_size"] = 256 // 4
# trainer_config["epochs"] = 100
# trainer_config["shuffle"] = False
dp_config["eval_frequency"] = 10
dp_config["num_epocks"] = 100
dp_config["batch_size"] = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = 3


train_dataset = AttractorTrajectoryDataset(
    pred_horizon=dp_config["pred_horizon"],
    obs_horizon=dp_config["obs_horizon"],
    action_horizon=dp_config["action_horizon"],
    state_dim=input_dim,
    num_trajectories=200,
)

logger.info(f"{train_dataset=}")

valid_dataset = AttractorTrajectoryDataset(
    pred_horizon=dp_config["pred_horizon"],
    obs_horizon=dp_config["obs_horizon"],
    action_horizon=dp_config["action_horizon"],
    state_dim=input_dim,
    num_trajectories=20,
)

logger.info(f"{valid_dataset=}")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=dp_config["batch_size"],
    num_workers=dp_config["num_workers"],
    shuffle=True,
    pin_memory=True,
    persistent_workers=False,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=dp_config["batch_size"],
    num_workers=dp_config["num_workers"],
    shuffle=True,
    pin_memory=True,
    persistent_workers=False,
)

dp_config["obs_horizon"] = 2
dp_config["obs_dim"] = 3
dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

# Build models
dp_models = {}
dp_models["model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)
ema = dp.EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
dp_models["ema_model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)

# Create agent (assuming you have appropriate models)
# You'll need to provide actual model instances here
agent = DiffusionPolicy(models=dp_models, ema=ema, config=dp_config)

epochs, avg_losses, val_losses = [], [], []


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


# Create supervised trainer
trainer = SupervisedTrainer(
    agent=agent,
    trainer_config=dp_config,
    train_loader=train_loader,
    valid_loader=valid_loader,
    callback_fn=_cb,
)

# Train the model
trainer.train()

# # Evaluate the model
# trainer.eval()
