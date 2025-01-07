import os
from torch.utils.data import Dataset
from PIL import Image


# Build and load the MiniImageNet Dataset
class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.num_classes = 0
        self.mode = mode
        self.targets = []

        for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                continue
            self.num_classes += 1
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
                self.targets.append(class_idx)

    def __len__(self):
        return len(self.samples)

    # Overload the __getitem__ method of the DataLoader class.
    # If the dataset is initialized in train mode, the __getitem__ method will return two augmented views of the
    # same image.
    # If the dataset is initialized in eval mode, the __getitem__ method will return an augmented image and its label

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            if self.mode == "train":
                aug1 = self.transform(image)
                aug2 = self.transform(image)
                return aug1, aug2

            elif self.mode == "eval":
                image = self.transform(image)
                return image, label
