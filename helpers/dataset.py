import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, Pad
from torchvision.datasets import CIFAR10 # Added cifar10
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import tqdm
import math
from torchvision.utils import save_image
import os

class dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.SEED = 0
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        print(f"Seed: {self.SEED}")

        transform = Compose([
            ToTensor(),
           ]
        )

        dataset_path = '../datasets'  # Relative path from FlowOT_UNet to structured_code/datasets
        full_path = os.path.abspath(dataset_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        print(f"Dataset will be downloaded to: {full_path}")

        self.dataset = CIFAR10(dataset_path, download=True, train=True, transform=transform)
        self.dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        self.show_first_batch(self.dataloader)

    def show_images(self, images, title=""):
        images = images.detach().cpu().numpy()  # Convert images to numpy arrays
        fig = plt.figure(figsize=(4, 4))
        cols = math.ceil(len(images) ** (1 / 2))
        rows = math.ceil(len(images) / cols)

        for r in range(rows):
            for c in range(cols):
                idx = cols * r + c
                ax = fig.add_subplot(rows, cols, idx + 1)
                ax.axis('off')
                if idx < len(images):
                    # Transpose the image dimensions from (C, H, W) to (H, W, C)
                    img = images[idx].transpose((1, 2, 0))
                    # Display the image without applying a color map
                    ax.imshow(img)
        fig.suptitle(title, fontsize=18)
        plt.show()

    def show_first_batch(self, dataloader):
        for batch in dataloader:
            print(batch[0].shape)
            self.show_images(batch[0], "Images in the first batch")
            break
