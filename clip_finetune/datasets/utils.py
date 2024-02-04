import os
import os.path
from collections import Counter
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import warnings


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        kept_classes: Optional[List[int]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        if kept_classes is not None and class_index not in kept_classes:
            continue
        target_dir = os.path.join(directory, target_class)
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
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        image_loader (callable): A function to load a sample given its path.
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
            image_loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            env_label: Optional[int] = None,
            kept_classes: Optional[List[int]] = None,
            custom_env_labels: Optional[List[int]] = None,
            feature_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, kept_classes)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            # print out warning without interrupting the program
            warnings.warn(msg)

        if image_loader is None:
            image_loader = lambda x: x
        self.image_loader = image_loader

        self.extensions = extensions

        self.classes = classes
        self.num_classes = len(classes)
        self.class_to_idx = class_to_idx
        self.n_classes = len(classes)
        self.samples = samples
        self.targets = np.array([s[1] for s in samples])
        self.env_label = env_label
        self.custom_env_labels = custom_env_labels
        self.feature_data = feature_data

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def get_env_label(self, index):
        if self.env_label is not None:
            env_label = self.env_label if self.custom_env_labels is None else self.custom_env_labels[index]
            return env_label
        else:
            return None

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, Any],]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.image_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        batch = {'images': sample, 'labels': target, 'image_paths': path, }
        if self.env_label is not None:
            if self.custom_env_labels is not None:
                raise NotImplementedError("Custom env labels not implemented for DomainBed")
                batch['metadata'] = self.custom_env_labels[index]
                # return sample, target, self.custom_env_labels[index]
            else:
                batch['metadata'] = self.env_label
        if self.feature_data is not None:
            batch['features'] = self.feature_data[path]
        return batch

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        image_loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            image_loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            env_label: Optional[int] = None,
            kept_classes: Optional[List[int]] = None,
            custom_env_labels: Optional[List[int]] = None,
            feature_data: Optional[Dict[str, Any]] = None,
    ):
        super(ImageFolder, self).__init__(root, image_loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, env_label=env_label,
                                          kept_classes=kept_classes, custom_env_labels=custom_env_labels,
                                          feature_data=feature_data)
        self.imgs = self.samples


def make_weights(sample_groups: np.ndarray, ):
    assert len(sample_groups.shape) == 1
    counts = Counter(sample_groups)

    n_groups = len(counts)
    n_samples = len(sample_groups)

    weights = np.zeros(n_samples)
    for i, (group, count) in enumerate(counts.items()):
        weights[sample_groups == group] = n_samples / (n_groups * count)

    assert np.all(weights > 0)
    assert np.allclose(weights.sum(), n_samples, rtol=1e-5)
    return weights


def check_random_number_generator(seed: int = None, rng: np.random.RandomState = None):
    assert (seed is None) != (rng is None)
    if seed is not None:
        rng = np.random.RandomState(seed)
    assert isinstance(rng, np.random.RandomState)
    return rng


def make_env_class_masks(n_env: int, n_class: int, random_mask_ratio: float = None, unseen_class_ratio: float = None,
                        seed: int = None, rng: np.random.RandomState = None) -> dict:
    # entry = 1 means not masked (used in training)
    # entry = 0 means masked (used for OOD test)
    # only one of seed and rng can be not None
    rng = check_random_number_generator(seed, rng)
    all_masks = {}
    assert random_mask_ratio is None or unseen_class_ratio is None
    if random_mask_ratio is not None and random_mask_ratio > 0:
        mask = np.zeros((n_env, n_class), dtype=int)
        # make sure each env row has at least one True
        # make sure each class column has at least one True
        for i in range(n_env):
            mask[i, rng.choice(n_class)] = 1

        for j in range(n_class):
            if np.sum(mask[:, j]) == 0:
                mask[rng.choice(n_env), j] = 1

        n_entries = int(n_env * n_class * (1 - random_mask_ratio))
        assert n_entries > 0 and n_entries < n_env * n_class
        while np.sum(mask) < n_entries:
            i = rng.choice(n_env)
            j = rng.choice(n_class)
            mask[i, j] = 1
        assert np.sum(mask, axis=0).min() > 0
        assert np.sum(mask, axis=1).min() > 0
    elif unseen_class_ratio is not None and unseen_class_ratio > 0:
        mask = np.ones((n_env, n_class), dtype=int)
        n_unseen_class = int(n_class * unseen_class_ratio)
        assert n_unseen_class > 0 and n_unseen_class < n_class
        unseen_class = rng.choice(n_class, size=n_unseen_class, replace=False)
        mask[:, unseen_class] = 0
    else:
        raise ValueError(
            f'random_mask_ratio(={random_mask_ratio}) or unseen_class_ratio(={unseen_class_ratio}) must be > 0')

    all_masks['train'] = mask
    ood_masks = split_ood_mask_entries(mask, rng=rng)
    all_masks.update(ood_masks)
    return all_masks


def split_ood_mask_entries(mask, val_train_max_ratio=0.1, val_test_max_ratio=1.0, val_frac=0.25,
                           seed: int = None, rng: np.random.RandomState = None):
    """
    Splits the OOD entries of a given env-class mask into validation and test sets.

    The function splits the OOD entries of the mask into validation and test sets
    while maintaining the following constraints:
    - val_train_max_ratio * #ID entries >= #validation entries
    - val_test_max_ratio * #validation entries >= #test entries
    - #validation entries are set initially to val_frac * #ID entries

    Args:
        mask (np.array): A binary matrix where 1 denotes ID entry and 0 denotes OOD entry.
        val_train_max_ratio (float, optional): Maximum ratio of validation entries to ID entries.
                                               Defaults to 0.1.
        val_test_max_ratio (float, optional): Maximum ratio of test entries to validation entries.
                                              Defaults to 1.0.
        val_frac (float, optional): Initial fraction of ID entries to determine the number of
                                     validation entries. Defaults to 0.25.

    Returns:
        dict: A dictionary containing two masks, "val" for validation set and "test"
              for the test set.
    """
    # Calculate the number of ID and OOD entries
    rng = check_random_number_generator(seed, rng)

    num_id_entries = int(np.sum(mask))
    num_ood_entries = int(np.prod(mask.shape) - num_id_entries)

    # Calculate the initial number of validation entries
    num_val_entries = int(val_frac * num_id_entries)

    # Calculate the upper and lower bounds for validation entries
    n_val_max_1 = int(val_train_max_ratio * num_id_entries)
    n_val_max_2 = int(num_ood_entries * val_test_max_ratio / (1 + val_test_max_ratio))
    n_val_min, n_val_max = 1, max(min(n_val_max_1, n_val_max_2), 1)

    # Clip the number of validation entries to be within the specified bounds
    num_val_entries = np.clip(num_val_entries, n_val_min, n_val_max)

    # Calculate the remaining OOD entries for the test set
    num_test_entries = num_ood_entries - num_val_entries
    assert num_test_entries > 0, f"Number of test entries (={num_test_entries}) must be positive."
    assert num_val_entries > 0, f"Number of validation entries (={num_val_entries}) must be positive."
    # Initialize the validation and test masks
    val_mask = np.zeros_like(mask)
    test_mask = np.zeros_like(mask)

    # Get the indices of the OOD entries
    ood_indices = np.argwhere(mask == 0)

    # Shuffle the OOD indices with rng
    rng.shuffle(ood_indices)

    # Split the OOD indices into validation and test sets
    val_indices = ood_indices[:num_val_entries]
    test_indices = ood_indices[num_val_entries:num_val_entries + num_test_entries]

    # Fill in the validation and test masks with the corresponding indices
    for i, j in val_indices:
        val_mask[i, j] = 1

    for i, j in test_indices:
        test_mask[i, j] = 1

    return {"val": val_mask, "test": test_mask}
