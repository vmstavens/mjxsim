import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from skrl.memories.torch import RandomMemory
from torch.utils.data import Subset

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)


def split_dataset(
    dataset: torch.utils.data.Dataset, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Splits a PyTorch Dataset into train and validation subsets.

    Args:
        dataset: An instance of a Dataset (e.g., PushTStateDataset).
        val_ratio: The fraction of the dataset to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_subset, val_subset) as Subset objects.
    """
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    # Ensure reproducible shuffling
    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = torch.utils.data.random_split(
        dataset, lengths=[train_len, val_len], generator=generator
    )
    return train_subset, val_subset


class DataHandler(pd.DataFrame):
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        memory: Optional["RandomMemory"] = None,
    ):
        super().__init__()
        self.memory = memory
        self.memories = [memory] if memory else []
        self.df = df
        self.dfs = [df] if df is not None else []
        self._file_paths = []
        self._reward_fn = None
        self._done_fn = None

    # ----------------------------
    # Public interface
    # ----------------------------

    def from_json(
        self, path: Union[str, Path], state_labels: list[str], action_labels: list[str]
    ) -> "DataHandler":
        states, actions = self._load_json_data(path)
        self.df = pd.DataFrame(
            np.hstack([states, actions]), columns=state_labels + action_labels
        )
        self.dfs.append(self.df)
        self._file_paths.append(path)
        return self

    def from_csv(
        self, path: Union[str, Path], state_labels: list[str], action_labels: list[str]
    ) -> "DataHandler":
        states, actions = self._load_csv_data(path, state_labels, action_labels)
        self.df = pd.DataFrame(
            np.hstack([states, actions]), columns=state_labels + action_labels
        )
        self.dfs.append(self.df)
        self._file_paths.append(path)
        return self

    def from_memory(self, memory: Union[str, "RandomMemory"]) -> "DataHandler":
        if isinstance(memory, str):
            memory = self._json_to_memory(memory)
        self.memory = memory
        self.memories.append(memory)
        tensor_names = memory.get_tensor_names()
        arrays = [
            memory.get_tensor_by_name(name).cpu().numpy() for name in tensor_names
        ]

        # Flatten each array to 2D
        arrays = [arr.reshape(arr.shape[0], -1) for arr in arrays]

        # Create DataFrame
        self.df = pd.DataFrame(np.hstack(arrays), columns=tensor_names)
        self.dfs.append(self.df)
        return self

    def from_folder(
        self,
        path: Union[str, Path],
        state_labels: list[str],
        action_labels: list[str],
        ext: str = ".csv",
    ) -> "DataHandler":
        memory = self._folder_to_memory(
            path,
            file_extension=ext,
            reward_func=self._reward_fn,
            done_func=self._done_fn,
            tensor_columns={"states": state_labels, "actions": action_labels},
        )
        return self.from_memory(memory)

    def trim(self, eps: float = 1e-3) -> "DataHandler":
        if self.df is None or self.df.empty:
            return self
        n_states = len(self.df.columns) // 2
        states = self.df.iloc[:, :n_states].to_numpy()
        actions = self.df.iloc[:, n_states:].to_numpy()
        states_trim, actions_trim = self._trim_constant_prefix(states, actions, eps)
        self.df = pd.DataFrame(
            np.hstack([states_trim, actions_trim]), columns=self.df.columns
        )
        self.dfs.append(self.df)
        return self

    @property
    def reward_fn(self) -> Callable:
        return self._reward_fn

    @reward_fn.setter
    def reward_fn(self, new_reward_fn: Callable) -> None:
        self._reward_fn = new_reward_fn

    @property
    def done_fn(self) -> Callable:
        return self._done_fn

    @done_fn.setter
    def done_fn(self, new_done_fn: Callable) -> None:
        self._done_fn = new_done_fn

    # ----------------------------
    # Internal helper functions
    # ----------------------------

    @staticmethod
    def _load_json_data(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        with open(path, "r") as f:
            data = json.load(f)
        states = np.array(data["states"], dtype=np.float32)
        actions = np.array(data["actions"], dtype=np.float32)
        return states, actions

    @staticmethod
    def _load_csv_data(
        path: Union[str, Path], state_labels: list[str], action_labels: list[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        states = df[state_labels].to_numpy(dtype=np.float32)
        actions = df[action_labels].to_numpy(dtype=np.float32)
        return states, actions

    @staticmethod
    def _trim_constant_prefix(
        states: np.ndarray, actions: np.ndarray, eps: float = 1e-3
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(states) == 0:
            return states, actions
        diffs = np.linalg.norm(np.diff(states, axis=0), axis=1)
        moving_indices = np.where(diffs > eps)[0]
        if len(moving_indices) == 0:
            return states, actions
        start_idx = moving_indices[0]
        return states[start_idx:], actions[start_idx:]

    @staticmethod
    def _json_to_memory(
        file_path: Union[str, Path], device: str = "cuda"
    ) -> "RandomMemory":
        with open(file_path, "r") as f:
            data = json.load(f)
        tensor_labels = list(data.keys())
        n_samples = np.shape(data[tensor_labels[0]])[0]
        memory = RandomMemory(memory_size=n_samples, device=device)
        samples_dict = {}
        for tensor_label in tensor_labels:
            tensor_data = np.array(data[tensor_label])
            if len(tensor_data.shape) > 2:
                tensor_data = tensor_data.squeeze()
            tensor_dim = tensor_data.shape[1] if len(tensor_data.shape) > 1 else 1
            memory.create_tensor(name=tensor_label, size=tensor_dim)
            samples_dict[tensor_label] = torch.FloatTensor(tensor_data).to(device)
        memory.add_samples(**samples_dict)
        return memory

    @staticmethod
    def _folder_to_memory(
        folder_path: Union[str, Path],
        file_extension: str = ".json",
        device: str = "cuda",
        reward_func: Optional[Callable] = None,
        done_func: Optional[Callable] = None,
        tensor_columns: Optional[Dict[str, list]] = None,
    ) -> "RandomMemory":
        folder_path = Path(folder_path)
        files = list(folder_path.glob(f"*{file_extension}"))
        if not files:
            raise FileNotFoundError(f"No {file_extension} files found in {folder_path}")

        all_data = []
        for f in files:
            if file_extension == ".json":
                data = json.load(open(f, "r"))
            else:  # CSV
                if tensor_columns is None:
                    raise ValueError("tensor_columns is required for CSV files")
                df = pd.read_csv(f)
                data = {
                    name: df[cols].values.astype(np.float32)
                    for name, cols in tensor_columns.items()
                }
            all_data.append(data)

        # Flatten all data
        total_samples = sum(len(list(d.values())[0]) for d in all_data)
        memory = RandomMemory(memory_size=total_samples, device=device)
        first_sample = all_data[0]
        for key, val in first_sample.items():
            size = val.shape[1] if len(val.shape) > 1 else 1
            memory.create_tensor(key, size)

        # Add all samples
        for data in all_data:
            batch = {k: torch.FloatTensor(v).to(device) for k, v in data.items()}
            memory.add_samples(**batch)

        return memory


def load_data(
    file_path: Union[str, Path],
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Load states and actions from a JSON file.

    Example JSON file structure:
    {
        "states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "actions": [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
    }

    Args:
        file_path (Union[str, Path]): Path to a `.json` file.

    Returns:
        Tuple[List[List[float]], List[List[float]]]:
            - states: List of states, e.g. [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            - actions: List of actions, e.g. [[0.01, 0.02], [0.03, 0.04]]
    """
    with open(file_path, "r") as f:
        data: Dict[str, List[List[float]]] = json.load(f)

    return data["states"], data["actions"]


def load_json(
    file_path: Union[str, Path],
) -> dict:
    """
    Load states and actions from a JSON file.

    Example JSON file structure:
    {
        "states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "actions": [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]
    }

    Args:
        file_path (Union[str, Path]): Path to a `.json` file.

    Returns:
        dict[str, List[List[float]]:
    """
    with open(file_path, "r") as f:
        data: Dict[str, List[List[float]]] = json.load(f)

    return data


def load_data_files(
    folder_path: Union[Path, str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load multiple episodes from a folder of JSON files.

    Example folder structure:
    data_folder/
        episode_1.json
        episode_2.json
        episode_3.json

    Example output:
    data = {
        "states": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
        "actions": np.array([[0.01], [0.02], [0.03]], dtype=np.float32)
    }
    episode_indices = np.array([2, 3], dtype=np.int64)  # Episode ends at indices 2, 3

    Args:
        folder_path (Union[Path, str]): Path to a folder containing `.json` files.

    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray]:
            - data: {"states": np.ndarray, "actions": np.ndarray}
            - episode_indices: Array of cumulative episode end indices
    """
    folder_path_str = (
        folder_path if isinstance(folder_path, str) else folder_path.as_posix()
    )
    data: Dict[str, List[Any]] = {"states": [], "actions": []}
    episode_indices: List[int] = []
    end_pointer: int = 0

    files: List[str] = sorted(os.listdir(folder_path_str))
    for f in files:
        if f.endswith(".json"):
            file_path: str = os.path.join(folder_path_str, f)
            states, actions = load_data(file_path)
            num_data: int = len(states)
            end_pointer += num_data
            data["states"].extend(states)
            data["actions"].extend(actions)
            episode_indices.append(end_pointer)
    data["states"] = np.array(data["states"], dtype=np.float32)
    data["actions"] = np.array(data["actions"], dtype=np.float32)

    return data, np.array(episode_indices, dtype=np.int64)


def split_data(
    folder_path: Union[Path, str],
    shuffle: bool = True,
    train_split: float = 0.8,
    overwrite: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Split a folder of dataset files into `train` and `valid` subfolders.

    Example input folder structure:
    dataset/
        episode_1.json
        episode_2.json
        episode_3.json
        episode_4.json
        episode_5.json

    Example output structure:
    dataset/
        train/
            episode_1.json
            episode_3.json
            episode_4.json
        valid/
            episode_2.json
            episode_5.json

    Returns:
        tuple[list[str], list[str]]:
            - train_files: e.g. ['/path/dataset/episode_1.json', '/path/dataset/episode_3.json']
            - valid_files: e.g. ['/path/dataset/episode_2.json', '/path/dataset/episode_5.json']

    Raises:
        ValueError: If `train/` or `valid/` already exist and overwrite is False.
    """
    folder_path = folder_path if isinstance(folder_path, Path) else Path(folder_path)

    train_dir = folder_path / "train"
    valid_dir = folder_path / "valid"

    # Check if directories already exist
    if train_dir.exists() or valid_dir.exists():
        if not overwrite:
            raise ValueError(
                "Train or valid directories already exist. Set overwrite=True to replace them."
            )
        else:
            # Remove existing directories and their contents
            if train_dir.exists():
                shutil.rmtree(train_dir)
                logger.info(f"Removed existing directory: {train_dir}")
            if valid_dir.exists():
                shutil.rmtree(valid_dir)
                logger.info(f"Removed existing directory: {valid_dir}")

    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Get all files (excluding directories)
    file_names: List[str] = [
        (folder_path / item).as_posix()
        for item in os.listdir(folder_path)
        if (folder_path / item).is_file()
    ]

    if shuffle:
        random.shuffle(file_names)

    split_index = int(train_split * len(file_names))
    train_files = file_names[:split_index]
    valid_files = file_names[split_index:]

    # Copy files to respective directories
    for train_file in train_files:
        shutil.copy2(train_file, train_dir / Path(train_file).name)

    for valid_file in valid_files:
        shutil.copy2(valid_file, valid_dir / Path(valid_file).name)

    logger.info(
        f"Split completed: {len(train_files)} training files, {len(valid_files)} validation files"
    )

    return train_files, valid_files


def load_tcp_data(
    file_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load TCP pose data from a CSV file and compute states and actions.

    Example CSV columns:
        actual_TCP_pose_0, actual_TCP_pose_1, ..., actual_TCP_pose_5,
        target_TCP_pose_0, target_TCP_pose_1, ..., target_TCP_pose_5

    Example output:
        states = np.array([
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],  # First timestep actual pose
            [0.2, 0.3, 0.4, 0.1, 0.0, 0.0]   # Second timestep actual pose
        ], dtype=np.float32)

        actions = np.array([
            [0.01, 0.02, 0.03, 0.0, 0.0, 0.0],  # target - actual for first timestep
            [0.02, 0.03, 0.04, 0.1, 0.0, 0.0]   # target - actual for second timestep
        ], dtype=np.float32)

    Args:
        file_path (Union[str, Path]): Path to the CSV file containing robot logs.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - states: (N, 6) array of actual TCP poses [x, y, z, rx, ry, rz]
            - actions: (N, 6) array of delta TCP poses (target - actual)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load CSV
    df = pd.read_csv(file_path)

    # Extract actual TCP pose (state)
    state_cols = [f"actual_TCP_pose_{i}" for i in range(6)]
    states = df[state_cols].to_numpy(dtype=np.float32)

    # Extract target TCP pose
    target_cols = [f"target_TCP_pose_{i}" for i in range(6)]
    targets = df[target_cols].to_numpy(dtype=np.float32)

    # Compute action as delta (target - actual)
    actions = targets - states

    return states, actions


def trim_constant_prefix(
    states: np.ndarray, actions: np.ndarray, eps: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the initial constant segment from trajectory data.

    Example input:
        states = np.array([
            [0.0, 0.0, 0.0],  # Constant segment
            [0.0, 0.0, 0.0],  # Constant segment
            [0.1, 0.2, 0.3],  # Motion starts here
            [0.2, 0.3, 0.4]
        ])
        actions = np.array([[0], [0], [1], [1]])

    Example output (with eps=0.001):
        states = np.array([
            [0.1, 0.2, 0.3],  # Motion segment kept
            [0.2, 0.3, 0.4]
        ])
        actions = np.array([[1], [1]])

    Args:
        states: Array of state trajectories with shape (N, state_dim)
        actions: Array of action trajectories with shape (N, action_dim)
        eps: Threshold for detecting motion (L2 norm of state differences)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Trimmed states and actions arrays
    """
    if len(states) == 0:
        return states, actions

    # Compute per-step change in states
    diffs = np.linalg.norm(np.diff(states, axis=0), axis=1)

    # Debug: print some statistics
    logger.info(
        f"Diff stats: min={diffs.min():.6f}, max={diffs.max():.6f}, mean={diffs.mean():.6f}"
    )
    logger.info(f"First 10 diffs: {diffs[:10]}")
    logger.info(f"Using eps={eps}")

    # Find first index where motion exceeds eps
    moving_indices = np.where(diffs > eps)[0]

    logger.info(f"Found {len(moving_indices)} points with motion > eps")

    if len(moving_indices) == 0:
        logger.info("No motion detected above threshold")
        return states, actions

    # Start after the first detected motion step
    start_idx = moving_indices[0]
    logger.info(f"Trimming from index {start_idx} (keeping from {start_idx} to end)")

    if not isinstance(states, np.ndarray):
        states = np.array(states)
    if not isinstance(actions, np.ndarray):
        actions = np.array(actions)

    return states[start_idx:], actions[start_idx:]


def csv_to_json(
    file_path: Union[str, Path],
    state_labels: list[str],
    action_labels: list[str],
    write: bool = False,
) -> None:
    """
    Convert CSV file to JSON format for trajectory data.

    Example CSV structure:
        actual_TCP_pose_0, actual_TCP_pose_1, actual_TCP_pose_2, actual_TCP_speed_0, actual_TCP_speed_1, actual_TCP_speed_2
        0.1, 0.2, 0.3, 0.01, 0.02, 0.03
        0.2, 0.3, 0.4, 0.02, 0.03, 0.04

    Example usage:
        state_labels = [f"actual_TCP_pose_{i}" for i in range(3)]
        action_labels = [f"actual_TCP_speed_{i}" for i in range(3)]
        result = csv_to_json("data.csv", state_labels, action_labels, write=True)

    Example JSON output:
        {
            "states": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            "actions": [[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]]
        }

    Args:
        file_path: Path to CSV file
        state_labels: List of column names for state data
        action_labels: List of column names for action data
        write: Whether to write the result to a JSON file

    Returns:
        Dictionary with "states" and "actions" keys containing the converted data
    """
    # Read CSV file
    df = pd.read_csv(file_path)

    # Extract states and actions
    states = df[state_labels].values.tolist()
    actions = df[action_labels].values.tolist()

    # Create result structure
    result = {
        "states": states,
        "actions": actions,
    }

    # Create output file path in json folder
    file_path = Path(file_path)
    output_dir = file_path.parent / "json"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the same relative structure
    output_path = output_dir / file_path.with_suffix(".json").name

    if write:
        # Write to JSON file
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)
        logger.info(f"Converted {file_path} to {output_path}")

    return result


def json_to_memory(
    file_path: Union[str, Path],
    device: str = "cuda",
) -> RandomMemory:
    """
    Load trajectory data from a JSON file into a RandomMemory replay buffer.
    """
    # Load data from JSON
    data = load_json(file_path)

    tensor_labels = list(data.keys())
    n_samples = np.shape(data[list(data.keys())[0]])[0]

    logger.info(
        f"Found labels: {tensor_labels} in file {file_path} with {n_samples} samples."
    )

    # Set memory_size equal to actual number of samples
    memory_size = n_samples

    # Create memory buffer with exact size
    memory = RandomMemory(memory_size=memory_size, device=device)

    # Create tensors in memory
    samples_dict = {}
    for tensor_label in tensor_labels:
        tensor_data = data[tensor_label]

        # Fix the shape: remove extra dimensions
        tensor_data = np.array(tensor_data)
        if len(tensor_data.shape) > 2:
            # If shape is [N, 1, D], squeeze to [N, D]
            tensor_data = tensor_data.squeeze()

        tensor_dim = tensor_data.shape[1] if len(tensor_data.shape) > 1 else 1

        # Create tensor in memory
        memory.create_tensor(name=tensor_label, size=tensor_dim)

        # Convert to torch tensor
        samples_dict[tensor_label] = torch.FloatTensor(tensor_data).to(device)

    # Add all samples at once
    memory.add_samples(**samples_dict)

    logger.info(f"Loaded {len(memory)} samples from {file_path} into memory")

    # Debug: Check tensor shapes
    for name in memory.get_tensor_names():
        tensor = memory.get_tensor_by_name(name)
        print(f"Tensor '{name}' shape: {tensor.shape}")

    return memory


def csv_to_memory(
    file_path: Union[str, Path],
    tensor_columns: Dict[str, List[str]],
    device: str = "cuda",
) -> RandomMemory:
    """
    Flexible CSV to memory converter that can handle arbitrary tensors.
    """
    # Load CSV data
    df = pd.read_csv(file_path)

    # Get number of samples
    n_samples = len(df)

    # Set memory_size equal to actual number of samples
    memory_size = n_samples

    # Create memory buffer with exact size
    memory = RandomMemory(memory_size=memory_size, device=device)

    # Create tensors and prepare data
    samples_dict = {}

    for tensor_name, columns in tensor_columns.items():
        # Get data for this tensor
        tensor_data = df[columns].values.astype(np.float32)
        tensor_dim = len(columns)

        # Create tensor in memory
        memory.create_tensor(name=tensor_name, size=tensor_dim)

        # Convert to torch tensor
        samples_dict[tensor_name] = torch.FloatTensor(tensor_data).to(device)

    # Add all samples at once
    memory.add_samples(**samples_dict)

    logger.info(f"Loaded {len(memory)} samples from {file_path} into memory")

    # Debug: Check tensor shapes
    for name in memory.get_tensor_names():
        tensor = memory.get_tensor_by_name(name)
        print(f"Tensor '{name}' shape: {tensor.shape}")

    return memory


def folder_to_memory(
    folder_path: Union[str, Path],
    file_extension: str = ".json",
    device: str = "cuda",
    reward_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    done_func: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    **kwargs,
) -> RandomMemory:
    """
    Load multiple trajectory files from a folder into a RandomMemory replay buffer.
    """
    folder_path = Path(folder_path)

    # Find all files with the specified extension
    files = list(folder_path.glob(f"*{file_extension}"))
    if not files:
        raise FileNotFoundError(f"No {file_extension} files found in {folder_path}")

    logger.info(f"Found {len(files)} {file_extension} files in {folder_path}")

    # Aggregate all data first to fix shapes
    all_data = {}
    tensor_columns = kwargs.get("tensor_columns", None)

    for file_path in files:
        logger.info(f"Loading data from: {file_path}")

        if file_extension == ".json":
            file_data = load_json(file_path)
        else:  # CSV
            if tensor_columns is None:
                raise ValueError("tensor_columns is required for CSV files")
            df = pd.read_csv(file_path)
            file_data = {}
            for tensor_name, columns in tensor_columns.items():
                file_data[tensor_name] = df[columns].values.astype(np.float32)

        # Initialize all_data with first file's structure
        if not all_data:
            all_data = {key: [] for key in file_data.keys()}

        # Append data from current file (fix shapes)
        for key in file_data.keys():
            tensor_data = np.array(file_data[key])
            # Fix shape: remove extra dimensions
            if len(tensor_data.shape) > 2:
                tensor_data = tensor_data.squeeze()
            all_data[key].append(tensor_data)

    # Process each file separately to handle next_states correctly
    all_samples = []

    for file_idx in range(len(files)):
        file_data = {}
        for key in all_data.keys():
            file_data[key] = all_data[key][file_idx]

        # Get the number of samples in this file
        num_samples = len(file_data[list(file_data.keys())[0]])

        # Create samples for this file
        for i in range(num_samples):
            sample = {}

            # Add current state and action
            for key, data in file_data.items():
                sample[key] = data[i]

            # Add next_state (i+1 state from the same file, if not last sample)
            if "states" in file_data and i < num_samples - 1:
                sample["next_states"] = file_data["states"][i + 1]
            elif "states" in file_data:
                # For the last sample, next_state is the same as current state (terminal state)
                sample["next_states"] = file_data["states"][i]

            # Add reward
            if (
                reward_func is not None
                and "states" in file_data
                and "actions" in file_data
            ):
                state_tensor = torch.FloatTensor(file_data["states"][i])
                action_tensor = torch.FloatTensor(file_data["actions"][i])
                sample["rewards"] = (
                    reward_func(state_tensor, action_tensor).cpu().numpy()
                )
            else:
                # Default reward: 0.0
                sample["rewards"] = np.array(0.0, dtype=np.float32)

            # Add terminated/dones - 1 for last element, 0 for others
            if i == num_samples - 1:
                # Last sample of the file gets terminated = 1
                sample["terminated"] = np.array(1.0, dtype=np.float32)
            else:
                # All other samples get terminated = 0
                sample["terminated"] = np.array(0.0, dtype=np.float32)

            all_samples.append(sample)

    # Create memory with exact size
    memory = RandomMemory(memory_size=len(all_samples), device=device)

    # Create tensors based on the first sample
    if all_samples:
        first_sample = all_samples[0]
        for tensor_name, tensor_value in first_sample.items():
            # Handle scalar values and arrays
            if hasattr(tensor_value, "shape") and len(tensor_value.shape) > 0:
                tensor_dim = tensor_value.shape[0]
            else:
                tensor_dim = 1
            memory.create_tensor(name=tensor_name, size=tensor_dim)

    # Add samples to memory in batches for efficiency
    batch_size = 100  # Adjust based on your memory constraints
    for i in range(0, len(all_samples), batch_size):
        batch_samples = all_samples[i : i + batch_size]

        # Convert batch to tensors
        samples_dict = {}
        for tensor_name in first_sample.keys():
            tensor_data = []
            for sample in batch_samples:
                value = sample[tensor_name]
                # Ensure correct shape
                if not hasattr(value, "shape") or len(value.shape) == 0:
                    value = np.array([value])
                tensor_data.append(value)

            tensor_array = np.array(tensor_data)
            # Fix shape if needed
            if len(tensor_array.shape) == 1:
                tensor_array = tensor_array.reshape(-1, 1)

            samples_dict[tensor_name] = torch.FloatTensor(tensor_array).to(device)

        memory.add_samples(**samples_dict)

    logger.info(f"Loaded {len(memory)} total samples from {len(files)} files")

    return memory


if __name__ == "__main__":
    from pathlib import Path

    # Path to your folder of CSV files
    dataset_path_real = Path(__file__).parent.parent / "data/robotB_data_trimmed"

    # Define column labels in your CSV files
    state_labels = [f"actual_q_{i}" for i in range(6)]  # CSV columns for states
    action_labels = [f"actual_qd_{i}" for i in range(6)]  # CSV columns for actions

    # Map tensor names to column names
    tensor_columns = {
        "states": state_labels,
        "actions": action_labels,
    }

    goal = torch.Tensor(
        [-1.13769275, -1.84879365, -2.23483372, -0.61194004, 1.53735435, 0.48228303]
    )

    # Optional: define a reward function if needed
    def get_reward(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Example: simple negative L2 norm of actions
        return -torch.linalg.norm(state - goal)

    state_labels = ["actual_TCP_pose_0", "actual_TCP_pose_1", "actual_TCP_pose_2"]
    action_labels = ["actual_TCP_speed_0", "actual_TCP_speed_1", "actual_TCP_speed_2"]

    data_handler = DataHandler()
    data_handler.from_folder(
        dataset_path_real,
        state_labels=state_labels,
        action_labels=action_labels,
        ext=".csv",
    )

    # Trim initial constant segments
    data_handler.trim(eps=0.001)

    # Access the resulting dataframe
    df = data_handler.df
    print(df.head())
