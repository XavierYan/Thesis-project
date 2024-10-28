import torch
from torch.utils.data import Dataset, DataLoader
import pickle

import torch
from torch.utils.data import Dataset
import pickle

class FeatureDataset(Dataset):
    def __init__(self, pickle_file_path):
        # load pickle file
        with open(pickle_file_path, 'rb') as file:
            self.features = pickle.load(file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample_features = self.features[idx]


        history_trajectory = torch.tensor(sample_features['history_trajectory'], dtype=torch.float)
        reference_trajectory = torch.tensor(sample_features['reference_trajectory'], dtype=torch.float)
        nearby_trajectories = torch.tensor(sample_features['nearby_trajectories'], dtype=torch.float)
        nearby_reference_trajectories = torch.tensor(sample_features['nearby_reference_trajectories'], dtype=torch.float)
        # nearby_trajectories = torch.zeros((2,5,9))
        # nearby_reference_trajectories = torch.zeros((2,15,5))
        lane_boundaries = torch.tensor(sample_features['lane_boundaries'], dtype=torch.float)
        label = torch.tensor(sample_features['label'], dtype=torch.float)


        return {
            'history_trajectory': history_trajectory,
            'reference_trajectory': reference_trajectory,
            'nearby_trajectories': nearby_trajectories,
            'nearby_reference_trajectories': nearby_reference_trajectories,
            'lane_boundaries': lane_boundaries,
            'label': label
        }

# dataset = FeatureDataset("/home/xavier/project/thesis/src/features/train/training_features.pkl")

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)


# for batch in dataloader:
#     print("Batch data:")
#     for key, value in batch.items():
#         print(f"Shape of {key}: {value.shape}")  # Print the shape of each item
#         print(f"Data of {key}: {value}")         # Print the actual data of each item
#     break  # Only process one batch for inspection