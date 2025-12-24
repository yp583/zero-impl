import numpy as np
from torch.utils.data import Dataset

class MyRandomDataset(Dataset):
    def __init__(self, in_dim, out_dim, num_samples, seed=42):
        self.num_samples = num_samples
        self.data = np.random.randn(self.num_samples, in_dim)  # random data
        self.labels = np.random.randint(0, out_dim, size=(self.num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.labels[index]