import os
import torch
import shutil
import math
import datetime
import tqdm
import cddpm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10 # Added cifar10
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths  #pip install pytorch-fid

class FID:
    def __init__(self): 
        transform = Compose([
            ToTensor(),
            ]
        )
        self.dataset = CIFAR10("./test_dataset/cifar10/", download=True, train=False, transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        
    def calculate_FID_DDPM(self, model, cddpm, eval_batch_size: int, img_size: int, device):
        real_images_path = './FID/real_images/'
        image_batch_size = 20

        # Ensure the real images directory is clean
        if os.path.exists(real_images_path):
            shutil.rmtree(real_images_path)
        os.makedirs(real_images_path)

        # Saving real images
        print("Saving real images...")
        image_counter = 0  # Counter to keep track of how many images are saved
        for batch_idx, (images, _) in enumerate(self.dataloader):
            for image in images:
                if image_counter < eval_batch_size:  # Check if less than 10,000 images are saved
                    image_save_path = os.path.join(real_images_path, f'real_image_{image_counter}.png')
                    save_image(image, image_save_path)
                    image_counter += 1
                else:
                    break  # Stop the loop if 10,000 images have been saved
            if image_counter >= eval_batch_size:
                break

        # Setup for saving generated images
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        generated_images_path = f'./FID/generated_images_{current_time}/'
        os.makedirs(generated_images_path)
        print("Real images Saved, beginning Evaluation...")

        # Generating and saving synthetic images
        eval_range = math.ceil(eval_batch_size / image_batch_size)

        with torch.no_grad():
            for eval_epoch in range(eval_range):
                random_label_vector = torch.randint(0,10, (image_batch_size,)).to(device)
                sample = torch.randn((image_batch_size, 3, img_size, img_size)).to(device)
                for i in tqdm.tqdm(reversed(range(1, cddpm.num_timesteps)), position= 0):
                    t = (torch.ones(image_batch_size) * i).long().to(device)
                    predicted_noise = model(t, sample, random_label_vector)
                    sample = cddpm.step(predicted_noise, i, sample)
                output = sample.clamp(0,1)


                for i, image in enumerate(output):
                    save_image(image, os.path.join(generated_images_path, f'image_{eval_epoch*image_batch_size + i}.png'))

        # Calculating FID score
        fid_value = calculate_fid_given_paths([real_images_path, generated_images_path],
                                            batch_size=eval_batch_size,
                                            device=str(device),
                                            dims=2048)
        return fid_value
    
    def calculate_FID(self, model, eval_batch_size: int, img_size: int, device):
        real_images_path = './FID/real_images/'
        image_batch_size = 20

        # Ensure the real images directory is clean
        if os.path.exists(real_images_path):
            shutil.rmtree(real_images_path)
        os.makedirs(real_images_path)

        # Saving real images
        print("Saving real images...")
        image_counter = 0  # Counter to keep track of how many images are saved
        for batch_idx, (images, _) in enumerate(self.dataloader):
            for image in images:
                if image_counter < eval_batch_size:  # Check if less than 10,000 images are saved
                    image_save_path = os.path.join(real_images_path, f'real_image_{image_counter}.png')
                    save_image(image, image_save_path)
                    image_counter += 1
                else:
                    break  # Stop the loop if 10,000 images have been saved
            if image_counter >= eval_batch_size:
                break

        # Setup for saving generated images
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        generated_images_path = f'./FID/generated_images_{current_time}/'
        os.makedirs(generated_images_path)
        print("Real images Saved, beginning Evaluation...")

        # Generating and saving synthetic images
        eval_range = math.ceil(eval_batch_size / image_batch_size)
        with torch.no_grad():
            for eval_epoch in range(eval_range):
                random_label_vector = torch.randint(0, 10, (image_batch_size,)).to(device)
                sample = torch.randn((image_batch_size, 3, img_size, img_size)).to(device)
                output = model.decode(sample, random_label_vector).clamp(0, 1)

                for i, image in enumerate(output):
                    save_image(image, os.path.join(generated_images_path, f'image_{eval_epoch*image_batch_size + i}.png'))

        # Calculating FID score
        fid_value = calculate_fid_given_paths([real_images_path, generated_images_path],
                                            batch_size=eval_batch_size,
                                            device=str(device),
                                            dims=2048)
        return fid_value

