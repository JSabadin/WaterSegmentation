import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
from torchvision.datasets import VisionDataset
import random
import torch
from torchvision import transforms
import numpy as np


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    Modified for a dataset with a single class and no class subdirectories.
    """

    # Check if the directory exists
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Couldn't find the directory {directory}.")

    # Use an empty string as a placeholder for the single class name
    classes = [""]  # Single placeholder class
    class_to_idx = {classes[0]: 0}

    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    split_files: Optional[Dict[str, List[str]]] = None  # New parameter
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): Root dataset directory.
        class_to_idx (Dict[str, int], optional): Dictionary mapping class name to class index.
        extensions (Union[str, Tuple[str, ...]], optional): Allowed extensions.
        is_valid_file (Callable[[str], bool], optional): Function to check if file is valid.
        split_files (Dict[str, List[str]], optional): Dictionary for dataset splitting.

    Returns:
        List[Tuple[str, int]]: List of (sample path, class_index) tuples.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    if (extensions is None) == (is_valid_file is None):
        raise ValueError("Either extensions or is_valid_file must be provided, but not both")

    is_valid_file = is_valid_file or (lambda x: has_file_allowed_extension(x, extensions))

    instances = []
    if split_files is not None:
        for class_name, files in split_files.items():
            class_index = class_to_idx.get(class_name, 0)
            for file_name in files:
                file_path = file_name
                if is_valid_file(file_path):
                    item = file_path, class_index
                    instances.append(item)
    else:
        # Default behavior: list all files in the directory
        for class_name, class_index in class_to_idx.items():
            target_dir = os.path.join(directory, class_name)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        split_files: Optional[Dict[str, List[str]]] = None  # New parameter
    ) -> None:
        super().__init__(root, transform=transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file, split_files)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        split_files: Optional[Dict[str, List[str]]] = None  # New parameter
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, split_files=split_files)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __len__(self) -> int:
        return len(self.samples)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def binary_loader(path: str) -> Image.Image:
    blue_color = np.array([41, 167, 224])

    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")  # Convert to RGB to ensure we have three channels

        # Convert the image to a numpy array once
        data = np.array(img)

        # Use numpy broadcasting to create a mask
        mask = np.all(data == blue_color, axis=-1)

        # Directly convert the boolean mask to 'uint8' and multiply by 255
        data = np.where(mask[..., None], [255, 255, 255], [0, 0, 0]).astype(np.uint8)

        # Convert back to an image
        binary_img = Image.fromarray(data, 'RGB').convert('L')

        return binary_img

class ImageFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        mask_root: str,
        transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        mask_loader: Callable[[str], Any] = binary_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        is_train: bool = False,  # Flag to control the application of augmentations
        split_files: Optional[Dict[str, List[str]]] = None,  # New parameter for dataset splitting
        mean: Optional[List[float]] = None,  # New parameter for dataset normalization
        std: Optional[List[float]] = None  # New parameter for dataset normalization
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            is_valid_file=is_valid_file,
            split_files=split_files  # Pass split_files to the superclass
        )
        self.mask_root = mask_root
        self.mask_loader = mask_loader
        self.imgs = self.samples
        self.masks = self.make_masks(self.mask_root, self.imgs)
        self.is_train = is_train  # Store the flag
        self.mean = mean
        self.std = std
        # Define the transformations
        self.final_transforms = transform  # Convert PIL Image to tensor

    @staticmethod
    def make_masks(mask_root: str, imgs: List[Tuple[str, int]]) -> List[str]:
        mask_paths = []
        for img_path, _ in imgs:
            mask_path = mask_root + '\\' + os.path.basename(img_path.replace('.jpg', '.png').split('\\')[-1])
            if os.path.exists(mask_path):
                mask_paths.append(mask_path)
            else:
                mask_paths.append(None)
        return mask_paths

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        mask = self.mask_loader(self.masks[index]) if self.masks[index] else None

        # Apply the transformations only if is_train is True
        if self.is_train:
            angle = random.uniform(-0.01, 0.01)  # Select a random angle between -2 and 2
            sample = transforms.functional.rotate(sample, angle)  # Rotate the original image
            mask = transforms.functional.rotate(mask, angle)  # Rotate the mask

            if random.random() > 0.5:  # If a random number between 0 and 1 is larger than 0.5
                sample = transforms.functional.hflip(sample)  # Flip the original image
                mask = transforms.functional.hflip(mask)  # Flip the mask

            sample = transforms.functional.adjust_contrast(sample, random.uniform(0.8, 1.2))  # Apply random contrast change to the original image

        # Apply the final transformations
        sample = self.final_transforms(sample)
        sample = transforms.Normalize(mean=self.mean, std=self.std)(sample)  # Normalize the image!!!
        mask = self.final_transforms(mask) 

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, mask, path, target