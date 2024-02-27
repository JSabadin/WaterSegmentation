from ModifiedTorchvisionDatasetsFolder import ImageFolder
from torchvision.transforms import Compose, ToTensor
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
import json

def calculate_mean_std(path, dataset, imagesize):
    transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.ToTensor()
    ])

    dataset_path = os.path.join(path, dataset)
    data_samples = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, file))]

    sum_images = torch.tensor([0.0, 0.0, 0.0])
    sum_sq_images = torch.tensor([0.0, 0.0, 0.0])
    count = 0

    for sample in data_samples:
        img = Image.open(sample).convert('RGB')
        img = transform(img)
        sum_images += torch.mean(img, dim=[1,2])
        sum_sq_images += torch.mean(img**2, dim=[1,2])
        count += 1

    mean = sum_images / count
    std = (sum_sq_images / count - mean**2)**0.5

    return mean.tolist(), std.tolist()


def load_data(batch_size=4, imagesize=(224, 224)):
    # Calculate mean and standard deviation for normalization
    # mean, std = calculate_mean_std("./seminar_ST_2023_24", "RGB", imagesize)  # This is computationaly expensive so we just save the values
    mean = [0.39206787943840027, 0.4307292103767395, 0.39849820733070374]
    std = [0.2781929075717926, 0.28806477785110474, 0.31973788142204285]
    print("Calculated Mean:", mean)
    print("Calculated Standard Deviation:", std)

    # Define transformations for masks and the images! Normalisation is ommited, since it is only applied to the images and not the masks
    common_transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.ToTensor(),
    ])

    # Paths to your images and masks
    image_root = "./seminar_ST_2023_24/RGB"
    mask_root = "./seminar_ST_2023_24/WASR"


    # List all files and split into training, validation, and testing
    all_samples = sorted(os.listdir(image_root))
    train_samples = [os.path.join(image_root, f) for i, f in enumerate(all_samples) if i % 5 != 3 and i % 5 != 4]
    val_samples = [os.path.join(image_root, all_samples[i]) for i in range(len(all_samples)) if i % 5 == 3]
    test_samples = [os.path.join(image_root, all_samples[i]) for i in range(len(all_samples)) if i % 5 == 4]
    ######## ablation study #####################################################################
    # all_samples = sorted(os.listdir(image_root))
    # total_samples = len(all_samples)

    # # Calculate split indices
    # train_end = int(total_samples * 0.6)
    # val_end = train_end + int(total_samples * 0.2)

    # # Split the datasets
    # train_samples = [os.path.join(image_root, f) for f in all_samples[:train_end]]
    # val_samples = [os.path.join(image_root, f) for f in all_samples[train_end:val_end]]
    # test_samples = [os.path.join(image_root, f) for f in all_samples[val_end:]]
    #############################################################################################
    # Create instances of the modified ImageFolder class for both train and test datasets
    train_dataset = ImageFolder(
        root=image_root,
        mask_root=mask_root,
        transform=common_transform,
        is_train=True, # For setting the augmentations to increase the training data size!
        split_files  = {"": train_samples}, # "" stands for the class ID. Since we are doing segmentation, no class ID is needed.
        mean = mean,
        std = std,
    )

    val_dataset = ImageFolder(
        root=image_root, 
        mask_root=mask_root, 
        transform=common_transform, 
        split_files = {"": val_samples},
        mean = mean,
        std = std,
    )

    test_dataset = ImageFolder(
        root=image_root,
        mask_root=mask_root,
        transform=common_transform,
        split_files = {"": test_samples}, 
        mean = mean,
        std = std,
    )

    # Create DataLoaders for both train and test datasets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader

def denormalize(image, mean, std):
    image = image.clone().detach().cpu().numpy()  # Clone the tensor, detach it from the computation graph, and convert it to numpy
    image = image.transpose((1, 2, 0))  # Change shape from (C, H, W) to (H, W, C)
    image = image * std + mean  # Denormalize
    image = np.clip(image, 0, 1)  # Ensure values are within [0, 1] range
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer format
    return image


def show_images(images, masks, mean, std, num_images=4, show_normalised=False):
    actual_num_images = min(num_images, images.shape[0])  # In case the dataloader returns less than num_images
    for i in range(actual_num_images):
        # Denormalize and convert images and masks to numpy
        if show_normalised == False:
            img = denormalize(images[i], mean, std)
        else:
            img = images[i].cpu().numpy().transpose((1, 2, 0))
        
        mask = masks[i].cpu().numpy().squeeze()

        # Create an RGBA representation of the mask
        mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        mask_rgba[..., 0] = 0 # Red channel
        mask_rgba[..., 1] = 0.7  # Green channel
        mask_rgba[..., 2] = 1  # Blue channel
        mask_rgba[..., 3] = mask  # Alpha channel for transparency

        # Create a new figure for each pair of images
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Display the original image
        axs[0].imshow(img)
        axs[0].axis('off')  # Turn off axis

        # Display the original image with the mask overlay
        axs[1].imshow(img)
        axs[1].axis('off')  # Turn off axis
        axs[1].imshow(mask_rgba, cmap='gray', alpha=0.3)  # Overlay the mask with transparency

        # Remove space between the two images
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.show()
        


def load_whole_data(batch_size=4, imagesize=(224, 224)):
    # Calculate mean and standard deviation for normalization
    # mean, std = calculate_mean_std("./seminar_ST_2023_24", "RGB", imagesize)  # This is computationaly expensive so we just save the values
    mean = [0.39206787943840027, 0.4307292103767395, 0.39849820733070374]
    std = [0.2781929075717926, 0.28806477785110474, 0.31973788142204285]
    print("Calculated Mean:", mean)
    print("Calculated Standard Deviation:", std)

    # Define transformations for masks and the images! Normalisation is ommited, since it is only applied to the images and not the masks
    common_transform = transforms.Compose([
        transforms.Resize(imagesize),
        transforms.ToTensor(),
    ])

    # Paths to your images and masks
    image_root = "./seminar_ST_2023_24/RGB"
    mask_root = "./seminar_ST_2023_24/WASR"


    # List all files and split into training, validation, and testing
    all_samples = sorted(os.listdir(image_root))
    all_samples = [os.path.join(image_root, f) for i, f in enumerate(all_samples)]

    whole_dataset = ImageFolder(
        root=image_root, 
        mask_root=mask_root, 
        transform=common_transform, 
        split_files = {"": all_samples},
        mean = mean,
        std = std,
    )

    # Create DataLoaders for both train and test datasets
    whole_dataloader = torch.utils.data.DataLoader(whole_dataset, batch_size, shuffle=False, drop_last=False)

    return whole_dataloader

if __name__ == "__main__":
    batch_size=16
    original_imagesize = (414*3,736*3)# IMPORTANT: This is the original size of the images. You can find it in the dataset description.
    mean = [0.39206787943840027, 0.4307292103767395, 0.39849820733070374]
    std = [0.2781929075717926, 0.28806477785110474, 0.31973788142204285]
    train_dataloader, val_dataloader, test_dataloader = load_data(batch_size=batch_size, imagesize=(414,736))

    print(f"Length of train dataloader: {len(train_dataloader) * batch_size}")
    print(f"Length of validation dataloader: {len(val_dataloader) *  batch_size}")
    print(f"Length of test dataloader: {len(test_dataloader) * batch_size}")

    # Process the first 3 batches
    for i, (images, masks, targets) in enumerate(train_dataloader):
        print(f"Batch {i+1}:")
        print("Images shape:", images.shape)
        print("Masks shape:", masks.shape)

        # Show images and masks
        show_images(images, masks, mean=mean, std=std, show_normalised = True)

        # Break the loop after 3 batches
        if i == 2:
            break

