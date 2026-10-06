from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


CIFAR100_SUPERCLASS_HINT = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
]


@dataclass(frozen=True)
class A1DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_size: int
    val_size: int
    test_size: int
    image_size: int


class RemappedSubset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        class_to_new: dict[int, int],
        transform=None,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.class_to_new = class_to_new
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_new[int(target)]


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


def select_balanced_indices(
    targets: list[int],
    selected_classes: list[int],
    max_per_class: int | None,
) -> list[int]:
    counts = {cls: 0 for cls in selected_classes}
    selected: list[int] = []
    selected_set = set(selected_classes)
    for index, target in enumerate(targets):
        target = int(target)
        if target not in selected_set:
            continue
        if max_per_class is not None and counts[target] >= max_per_class:
            continue
        selected.append(index)
        counts[target] += 1
    return selected


def build_cifar100_subset_loaders(
    data_dir: str | Path,
    image_size: int,
    num_classes: int,
    max_train_per_class: int | None,
    max_test_per_class: int | None,
    val_ratio: float,
    batch_size: int,
    seed: int,
    num_workers: int,
    download: bool = True,
) -> A1DataBundle:
    root = Path(data_dir)
    raw_train = datasets.CIFAR100(root=root, train=True, download=download, transform=None)
    raw_test = datasets.CIFAR100(root=root, train=False, download=download, transform=None)

    selected_class_ids = list(range(num_classes))
    class_to_new = {class_id: new_id for new_id, class_id in enumerate(selected_class_ids)}
    class_names = [raw_train.classes[class_id] for class_id in selected_class_ids]
    train_indices = select_balanced_indices(raw_train.targets, selected_class_ids, max_train_per_class)
    test_indices = select_balanced_indices(raw_test.targets, selected_class_ids, max_test_per_class)

    train_transform, eval_transform = build_transforms(image_size)
    full_train = RemappedSubset(raw_train, train_indices, class_to_new, transform=train_transform)
    full_val = RemappedSubset(raw_train, train_indices, class_to_new, transform=eval_transform)
    test_set = RemappedSubset(raw_test, test_indices, class_to_new, transform=eval_transform)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset_indices = random_split(
        range(len(full_train)),
        [train_size, val_size],
        generator=generator,
    )
    train_set = Subset(full_train, list(train_subset.indices))
    val_set = Subset(full_val, list(val_subset_indices.indices))

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)

    return A1DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        train_size=train_size,
        val_size=val_size,
        test_size=len(test_set),
        image_size=image_size,
    )

