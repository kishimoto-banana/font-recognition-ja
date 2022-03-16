import random
from typing import Dict, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from transform import Transform


class FontDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        transform: Transform,
        font_labels: Dict[str, int],
        phase: str = "train",
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.phase = phase
        self.font_labels = font_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path).convert("L")

        if img.size[0] < 105:
            img = img.resize((105, 105))

        img_transformed = self.transform(np.array(img))
        patch_x = random.randint(0, img_transformed.shape[-1] - 105)
        patch = img_transformed[:, :, patch_x : patch_x + 105]

        font = image_path.split("/")[-1].split("_")[0]
        label = self.font_labels[font]

        return patch, label
