import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from mjxsim.agents.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    ConditionalUnet1D,
    DiffusionPolicy,
    EMAModel,
)
from mjxsim.datasets.attractor import AttractorTrajectoryDataset
from mjxsim.datasets.demonstration import DemonstrationDataset
from mjxsim.trainers.supervised_trainer import (
    SUPERVISED_TRAINER_DEFAULT_CONFIG,
    SupervisedTrainer,
)

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)

_HERE = Path(__file__).parent.parent
_SAVE_PATH = _HERE / f".demo/{Path(__file__).stem}/state/"
_SAVE_PATH.mkdir(parents=True, exist_ok=True)
_SAVE_PATH_MODELS = _HERE / ".demo/state/models/"
_SAVE_PATH_MODELS.mkdir(parents=True, exist_ok=True)
_SAVE_PATH_MEDIA = _HERE / ".demo/state/media/"
_SAVE_PATH_MEDIA.mkdir(parents=True, exist_ok=True)
dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG
trainer_config = SUPERVISED_TRAINER_DEFAULT_CONFIG
# model_path = Path(__file__).parent.parent / ".runs"
# model_path.mkdir(parents=True, exist_ok=True)

dp_config["experiment"]["directory"] = _SAVE_PATH_MODELS.as_posix()
dp_config["experiment"]["experiment_name"] = Path(__file__).stem
dp_config["experiment"]["wandb"] = True
# trainer_config["write_interval"] = 1
# trainer_config["checkpoint_interval"] = 10
# trainer_config["batch_size"] = 256 // 4
# trainer_config["epochs"] = 100
# trainer_config["shuffle"] = False
dp_config["eval_frequency"] = 10
dp_config["batch_size"] = 32
# dp_config["batch_size"] = 128
# dp_config["pred_horizon"] = 1
# dp_config["action_horizon"] = 16
dp_config["action_horizon"] = 1
dp_config["obs_horizon"] = 2

with open("data/pipe_insert/train/20251120_10_53_18.json", "r") as f:
    data0 = json.load(f)

obs_dim = np.array(data0["states"]).shape[1] # 6
act_dim = np.array(data0["actions"]).shape[1] # 4

pred_horizon   = dp_config["pred_horizon"]
obs_horizon    = dp_config["obs_horizon"]
action_horizon = dp_config["action_horizon"]

dp_config["obs_dim"] = obs_dim
dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

device = "cuda" if torch.cuda.is_available() else "cpu"

train_path = Path("data/pipe_insert/train")
valid_path = Path("data/pipe_insert/valid")

pct = 1.0

train_data_files = glob.glob(train_path.as_posix() + "/*.json")
valid_data_files = glob.glob(valid_path.as_posix() + "/*.json")

train_data_files = train_data_files[:int(pct*len(train_data_files))]
valid_data_files = valid_data_files[:int(pct*len(valid_data_files))]

pred_horizon = dp_config["pred_horizon"]
obs_horizon = dp_config["obs_horizon"]
action_horizon = dp_config["action_horizon"]

train_dataset = DemonstrationDataset(
    json_paths=train_data_files,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    reward_key=None,
    next_state_key=None,
    done_key=None,
)
valid_dataset = DemonstrationDataset(
    json_paths=valid_data_files,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
    reward_key=None,
    next_state_key=None,
    done_key=None,
)


logger.info(f"{train_dataset=}")

# logger.info(f"{valid_dataset=}")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=dp_config["batch_size"],
    num_workers=dp_config["num_workers"],
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=dp_config["batch_size"],
    num_workers=dp_config["num_workers"],
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
)

# Build models
dp_models = {}
dp_models["model"] = ConditionalUnet1D(act_dim, dp_config).to(device)
ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
dp_models["ema_model"] = ConditionalUnet1D(act_dim, dp_config).to(device)

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
