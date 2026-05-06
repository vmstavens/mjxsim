import json
import os
from pathlib import Path

import pandas as pd
import torch
from skrl.memories.torch import RandomMemory


class DataHandler(pd.DataFrame):
    """
    A subclass of pandas.DataFrame tailored for trajectory datasets
    with columns: 'states', 'actions', 'next_states', 'terminated', 'rewards'.
    """

    _metadata = ["dataframes_list", "combined_df"]

    @property
    def _constructor(self):
        return DataHandler

    # ============================================================
    # Instance Methods
    # ============================================================

    def trim_constant_prefix(self, eps=0.001):
        """Remove leading rows where states remain effectively constant."""
        if "states" not in self.columns or len(self) < 2:
            return self

        data = list(self["states"])
        first_state = data[0]
        start_index = 0
        for i in range(1, len(data)):
            diff = abs(torch.tensor(data[i]) - torch.tensor(first_state))
            if torch.any(diff > eps):
                start_index = i
                break
        return self.iloc[start_index:].reset_index(drop=True)

    def to_json_trajectory(self, write=False, output_dir="json/"):
        """Export the dataframe rows as JSON lines: each containing 'states' and 'actions'."""
        records = []
        for _, row in self.iterrows():
            records.append({"states": row["states"], "actions": row["actions"]})

        if write:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, "trajectory.json")
            with open(filename, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            return filename

        return records

    def to_random_memory(self, device="cuda"):
        """Convert the dataframe into a RandomMemory with states, actions, next_states, terminated, rewards."""
        df = self.copy()

        # Ensure required columns exist
        if "states" not in df.columns or "actions" not in df.columns:
            raise ValueError("DataFrame must contain 'states' and 'actions' columns.")

        # Autogenerate next_states if not present
        if "next_states" not in df.columns:
            next_states = df["states"].shift(-1)
            next_states.iloc[-1] = df["states"].iloc[-1]
            df["next_states"] = next_states

        # Autogenerate terminated
        if "terminated" not in df.columns:
            df["terminated"] = 0
            if len(df) > 0:
                df.at[len(df) - 1, "terminated"] = 1

        # Autogenerate rewards
        if "rewards" not in df.columns:
            df["rewards"] = 0.0

        # Create memory
        state_dim = len(df["states"].iloc[0])
        action_dim = len(df["actions"].iloc[0])
        memory = RandomMemory(memory_size=len(df), num_envs=1, device=device)
        memory.create_tensor("states", size=state_dim)
        memory.create_tensor("actions", size=action_dim)
        memory.create_tensor("next_states", size=state_dim)
        memory.create_tensor("terminated", size=1)
        memory.create_tensor("rewards", size=1)

        # Convert columns to tensors
        memory["states"].add(
            torch.tensor(df["states"].tolist(), dtype=torch.float32).to(device)
        )
        memory["actions"].add(
            torch.tensor(df["actions"].tolist(), dtype=torch.float32).to(device)
        )
        memory["next_states"].add(
            torch.tensor(df["next_states"].tolist(), dtype=torch.float32).to(device)
        )
        memory["terminated"].add(
            torch.tensor(df["terminated"].tolist(), dtype=torch.float32)
            .unsqueeze(-1)
            .to(device)
        )
        memory["rewards"].add(
            torch.tensor(df["rewards"].tolist(), dtype=torch.float32)
            .unsqueeze(-1)
            .to(device)
        )

        return memory

    # ============================================================
    # Class Methods
    # ============================================================

    @classmethod
    def from_json(cls, file_path):
        """Load a JSON lines file and return a TrajectoryDataFrame."""
        rows = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                rows.append(data)
        df = pd.DataFrame(rows)
        return cls(df)

    @classmethod
    def from_csv(cls, file_path, state_labels, action_labels):
        """Load CSV into a TrajectoryDataFrame, merging columns into 'states' and 'actions' arrays."""
        df_raw = pd.read_csv(file_path)
        df_raw["states"] = df_raw[state_labels].values.tolist()
        df_raw["actions"] = df_raw[action_labels].values.tolist()
        df = df_raw.drop(
            columns=[c for c in state_labels + action_labels if c in df_raw.columns]
        )
        return cls(df)

    @classmethod
    def from_folder(
        cls, folder_path, file_extension=".json", state_labels=None, action_labels=None
    ):
        """
        Load all files in a folder.
        Supports JSON or CSV (CSV requires state_labels and action_labels).
        Stores both a list of TrajectoryDataFrame and a combined DataFrame as attributes.
        """
        folder_path = os.path.abspath(folder_path)
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(file_extension.lower())
        ]

        dfs_list = []
        for file_path in files:
            if file_extension.lower() == ".json":
                dfs_list.append(cls.from_json(file_path))
            elif file_extension.lower() == ".csv":
                if state_labels is None or action_labels is None:
                    raise ValueError(
                        "state_labels and action_labels must be provided for CSV files"
                    )
                dfs_list.append(cls.from_csv(file_path, state_labels, action_labels))
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}")

        # Combine all into one DataFrame
        combined_df = pd.concat(dfs_list, ignore_index=True)
        combined_df = cls(combined_df)

        # Create an instance to hold attributes
        holder = combined_df.copy()
        holder.dataframes_list = dfs_list
        holder.combined_df = combined_df

        return holder


if __name__ == "__main__":
    file_path = (
        Path(__file__).parent.parent
        / "data/robotB_data_trimmed/21_08_2025_13_11_50_robot_B_data.csv"
    )
    folder_path = Path(__file__).parent.parent / "data/robotB_data_trimmed/trimmed"

    state_labels = [f"actual_q_{i}" for i in range(6)]
    action_labels = [f"actual_qd_{i}" for i in range(6)]

    dh = DataHandler.from_csv(
        file_path, state_labels=state_labels, action_labels=action_labels
    )

    dhs = DataHandler.from_folder("data/robotB_data_trimmed")

    print(dhs)
