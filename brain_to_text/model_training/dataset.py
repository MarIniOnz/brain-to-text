import math
import os
from typing import Optional, TypedDict

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class DayTrials(TypedDict):
    trials: list[int]
    session_path: str


class BrainToTextDataset(Dataset):
    """Dataset for brain-to-text data.

    Returns an entire batch of data instead of a single example
    """

    def __init__(
        self,
        trial_indices: dict[int, DayTrials],
        n_batches: int,
        split: str = "train",
        batch_size: int = 64,
        days_per_batch: int = 1,
        random_seed: int = -1,
        must_include_days=None,
        feature_subset=None,
    ) -> None:
        """
        Args:
            trial_indices (dict[int, DayTrials]): Dictionary mapping day numbers to dicts containing trial indices and session path.
            n_batches (int): Number of random training batches to create.
            split (str): Which split to use; either 'train' or 'test'.
            batch_size (int): Number of examples to include in each returned batch.
            days_per_batch (int): How many unique days can exist in a batch; controls per-day sampling.
            random_seed (int): Seed for reproducible random sampling. Use -1 for no fixed seed.
            must_include_days (list[int] | None): List of day indices that must be included in every batch, or None.
            feature_subset (list[int] | None): List of neural feature indices to keep from the input features, or None.
        """

        # Set random seed for reproducibility
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.split = split

        # Ensure the split is valid
        if self.split not in ["train", "test"]:
            raise ValueError(
                f'split must be either "train" or "test". Received {self.split}'
            )

        self.days_per_batch = days_per_batch

        self.batch_size = batch_size

        self.n_batches = n_batches

        self.days = {}
        self.n_trials = 0
        self.trial_indices = trial_indices
        self.n_days = len(trial_indices.keys())

        self.feature_subset = feature_subset

        # Calculate total number of trials in the dataset
        for d in trial_indices:
            self.n_trials += len(trial_indices[d]["trials"])

        if must_include_days is not None and len(must_include_days) > days_per_batch:
            raise ValueError(
                f"must_include_days must be less than or equal to days_per_batch. Received {must_include_days} and days_per_batch {days_per_batch}"
            )

        if (
            must_include_days is not None
            and len(must_include_days) > self.n_days
            and split != "train"
        ):
            raise ValueError(
                f"must_include_days is not valid for test data. Received {must_include_days} and but only {self.n_days} in the dataset"
            )

        if must_include_days is not None:
            # Map must_include_days to correct indices if they are negative
            for i, d in enumerate(must_include_days):
                if d < 0:
                    must_include_days[i] = self.n_days + d

        self.must_include_days = must_include_days

        # Ensure that the days_per_batch is not greater than the number of days in the dataset. Raise error
        if self.split == "train" and self.days_per_batch > self.n_days:
            raise ValueError(
                f"Requested days_per_batch: {days_per_batch} is greater than available days {self.n_days}."
            )

        if self.split == "train":
            self.batch_index = self.create_batch_index_train()
        else:
            self.batch_index = self.create_batch_index_test()
            self.n_batches = len(
                self.batch_index.keys()
            )  # The validation data has a fixed amount of data

    def __len__(self):
        """
        How many batches are in this dataset.
        Because training data is sampled randomly, there is no fixed dataset length,
        however this method is required for DataLoader to work
        """
        return self.n_batches

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Gets an entire batch of data from the dataset, not just a single item

        Args:
            idx (int): Index of the batch to retrieve

        Returns:
            batch (dict[str, torch.Tensor]): Dictionary containing the batch data with the following keys:
                'input_features' (torch.Tensor): Padded neural input features of shape (batch_size, max_time_steps, n_features)
                'seq_class_ids' (torch.Tensor): Padded phoneme class IDs of shape (batch_size, max_seq_len)
                'n_time_steps' (torch.Tensor): Number of time steps in each trial of shape (batch_size,)
                'phone_seq_lens' (torch.Tensor): Length of phoneme sequences
                'day_indices' (torch.Tensor): Day index for each trial of shape (batch_size,)
                'transcriptions' (torch.Tensor): Character level transcriptions of shape (batch_size,
                'block_nums' (torch.Tensor): Block numbers for each trial of shape (batch_size,)
                'trial_nums' (torch.Tensor): Trial numbers for each trial of shape (batch_size
        """
        batch = {
            "input_features": [],
            "seq_class_ids": [],
            "n_time_steps": [],
            "phone_seq_lens": [],
            "day_indices": [],
            "transcriptions": [],
            "block_nums": [],
            "trial_nums": [],
        }

        index = self.batch_index[idx]

        # Iterate through each day in the index
        for d in index.keys():
            # Open the hdf5 file for that day
            with h5py.File(self.trial_indices[d]["session_path"], "r") as f:
                # For each trial in the selected trials in that day
                for t in index[d]:
                    try:
                        g = f[f"trial_{t:04d}"]

                        # Remove features is neccessary
                        input_features = torch.from_numpy(
                            g["input_features"][:]
                        )  # neural data
                        if self.feature_subset:
                            input_features = input_features[:, self.feature_subset]

                        batch["input_features"].append(input_features)

                        batch["seq_class_ids"].append(
                            torch.from_numpy(g["seq_class_ids"][:])
                        )  # phoneme labels
                        batch["transcriptions"].append(
                            torch.from_numpy(g["transcription"][:])
                        )  # character level transcriptions
                        batch["n_time_steps"].append(
                            int(g.attrs["n_time_steps"])
                        )  # number of time steps in the trial - required since we are padding
                        batch["phone_seq_lens"].append(
                            g.attrs["seq_len"]
                        )  # number of phonemes in the label - required since we are padding
                        batch["day_indices"].append(
                            d
                        )  # day index of each trial - required for the day specific layers
                        batch["block_nums"].append(g.attrs["block_num"])
                        batch["trial_nums"].append(g.attrs["trial_num"])

                    except Exception as e:
                        print(
                            f"Error loading trial {t} from session {self.trial_indices[d]['session_path']}: {e}"
                        )
                        continue

        # Pad data to form a cohesive batch
        batch["input_features"] = pad_sequence(
            batch["input_features"], batch_first=True, padding_value=0
        )
        batch["seq_class_ids"] = pad_sequence(
            batch["seq_class_ids"], batch_first=True, padding_value=0
        )

        batch["n_time_steps"] = torch.tensor(batch["n_time_steps"])
        batch["phone_seq_lens"] = torch.tensor(batch["phone_seq_lens"])
        batch["day_indices"] = torch.tensor(batch["day_indices"])
        batch["transcriptions"] = torch.stack(batch["transcriptions"])
        batch["block_nums"] = torch.tensor(batch["block_nums"])
        batch["trial_nums"] = torch.tensor(batch["trial_nums"])

        return batch

    def create_batch_index_train(self) -> dict[str, np.ndarray[int]]:
        """Create an index that maps a batch_number to batch_size number of trials.

        Each batch will have days_per_batch unique days of data, with the number of trials for each day evenly split between the days
        (or as even as possible if batch_size is not divisible by days_per_batch)

        Returns:
            batch_index (dict[int, dict[int, np.ndarray[int]]]): Dictionary mapping batch index to a dictionary that maps day index
                to trial indices
        """

        batch_index = {}

        # Precompute the days that are not in must_include_days
        if self.must_include_days is not None:
            non_must_include_days = [
                d for d in self.trial_indices.keys() if d not in self.must_include_days
            ]

        for batch_idx in range(self.n_batches):
            batch = {}

            # Which days will be used for this batch. Picked randomly without replacement
            # TODO: In the future we may want to consider sampling days in proportion to the number of trials in each day

            # If must_include_days is not empty, we will use those days and then randomly sample the rest
            if self.must_include_days is not None and len(self.must_include_days) > 0:
                days = np.concatenate(
                    (
                        self.must_include_days,
                        np.random.choice(
                            non_must_include_days,
                            size=self.days_per_batch - len(self.must_include_days),
                            replace=False,
                        ),
                    )
                )

            # Otherwise we will select random days without replacement
            else:
                days = np.random.choice(
                    list(self.trial_indices.keys()),
                    size=self.days_per_batch,
                    replace=False,
                )

            # How many trials will be sampled from each day
            num_trials = math.ceil(
                self.batch_size / self.days_per_batch
            )  # Use ceiling to make sure we get at least batch_size trials

            for d in days:
                # Trials are sampled with replacement, so if a day has less than (self.batch_size / days_per_batch trials) trials, it won't be a problem
                trial_idxs = np.random.choice(
                    self.trial_indices[d]["trials"], size=num_trials, replace=True
                )
                batch[int(d)] = trial_idxs

            # Remove extra trials
            extra_trials = (num_trials * len(days)) - self.batch_size

            # While we still have extra trials, remove the last trial from a random day
            while extra_trials > 0:
                d = np.random.choice(days)
                batch[d] = batch[d][:-1]
                extra_trials -= 1

            batch_index[batch_idx] = batch

        return batch_index

    def create_batch_index_test(self) -> dict[int, dict[int, np.ndarray[int]]]:
        """Create an index that is all validation/testing data in batches of up to self.batch_size

        If a day does not have at least self.batch_size trials, then the batch size will be less than self.batch_size

        This index will ensures that every trial in the validation set is seen once and only once

        Returns:
            batch_index (dict[int, dict[int, np.ndarray[int]]]): Dictionary mapping batch index to a dictionary that maps day index
                to trial indices.
        """
        batch_index = {}
        batch_idx = 0

        for d in self.trial_indices.keys():
            # Calculate how many batches we need for this day
            num_trials = len(self.trial_indices[d]["trials"])
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size

            # Create batches for this day
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_trials)

                # Get the trial indices for this batch
                batch_trials = self.trial_indices[d]["trials"][start_idx:end_idx]

                # Add to batch_index
                batch_index[batch_idx] = {d: batch_trials}
                batch_idx += 1

        return batch_index


def train_test_split_indices(
    file_paths: list[str],
    test_percentage: float = 0.1,
    seed: int = -1,
    bad_trials_dict: Optional[dict] = None,
) -> tuple[dict[int, DayTrials], dict[int, DayTrials]]:
    """Split data from file_paths into train and test splits.

    Returns two dictionaries that detail which trials in each day will be a part of that split:
    Example:
        {
            0: trials[1,2,3], session_path: 'path'
            1: trials[2,5,6], session_path: 'path'
        }

    Args:
        file_paths (list[str]): List of file paths to the hdf5 files containing the data
        test_percentage (float): Percentage of trials to use for testing. 0 will use all trials for training, 1 will use all trials for testing
        seed (int): Seed for reproducibility. If set to -1, the split will be random
        bad_trials_dict (dict): Dictionary of trials to exclude from the dataset. Default is None. Formatted as:
            {
                'session_name_1': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                'session_name_2': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                ...
            }

    Returns:
        tuple[dict[int, DayTrials], dict[int, DayTrials]]: Two dictionaries containing the train and test trial indices for each day.
    """
    # Set seed for reporoducibility
    if seed != -1:
        np.random.seed(seed)

    # Get trials in each day
    trials_per_day = {}
    for i, path in enumerate(file_paths):
        session = [
            s
            for s in path.split("/")
            if (s.startswith("t15.20") or s.startswith("t12.20"))
        ][0]

        good_trial_indices = []

        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                num_trials = len(list(f.keys()))
                for t in range(num_trials):
                    key = f"trial_{t:04d}"

                    block_num = f[key].attrs["block_num"]
                    trial_num = f[key].attrs["trial_num"]

                    if (
                        bad_trials_dict is not None
                        and session in bad_trials_dict
                        and str(block_num) in bad_trials_dict[session]
                        and trial_num in bad_trials_dict[session][str(block_num)]
                    ):
                        # print(f'Bad trial: {session}_{block_num}_{trial_num}')
                        continue

                    good_trial_indices.append(t)

        trials_per_day[i] = {
            "num_trials": len(good_trial_indices),
            "trial_indices": good_trial_indices,
            "session_path": path,
        }

    # Pick test_percentage of trials from each day for testing and (1 - test_percentage) for training
    train_trials = {}
    test_trials = {}

    for day in trials_per_day.keys():
        num_trials = trials_per_day[day]["num_trials"]

        # Generate all trial indices for this day (assuming 0-indexed)
        all_trial_indices = trials_per_day[day]["trial_indices"]

        # If test_percentage is 0 or 1, we can just assign all trials to either train or test
        if test_percentage == 0:
            train_trials[day] = DayTrials(
                trials=all_trial_indices,
                session_path=trials_per_day[day]["session_path"],
            )
            test_trials[day] = DayTrials(
                trials=[], session_path=trials_per_day[day]["session_path"]
            )
            continue

        elif test_percentage == 1:
            train_trials[day] = DayTrials(
                trials=[], session_path=trials_per_day[day]["session_path"]
            )
            test_trials[day] = DayTrials(
                trials=all_trial_indices,
                session_path=trials_per_day[day]["session_path"],
            )
            continue

        else:
            # Calculate how many trials to use for testing
            num_test = max(1, int(num_trials * test_percentage))

            # Randomly select indices for testing
            test_indices = np.random.choice(
                all_trial_indices, size=num_test, replace=False
            ).tolist()

            # Remaining indices go to training
            train_indices = [
                idx for idx in all_trial_indices if idx not in test_indices
            ]

            # Store the split indices
            train_trials[day] = DayTrials(
                trials=train_indices, session_path=trials_per_day[day]["session_path"]
            )
            test_trials[day] = DayTrials(
                trials=test_indices, session_path=trials_per_day[day]["session_path"]
            )

    return train_trials, test_trials
