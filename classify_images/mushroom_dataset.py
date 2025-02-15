from torchvision import transforms
from torch.utils.data import Dataset
import random


class MushroomDataset(Dataset):
    def __init__(self, data_list, num_augmentations=1):
        self.data = data_list

        self.augmentation_list = [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]

        self.processed_data = []
        for item in data_list:
            original_image = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                ]
            )(item["image"])
            self.processed_data.append((original_image, item["encoded_genus"]))

            for _ in range(num_augmentations):
                # pick how many transforms for each image
                num_transforms = random.choice([1, 2])
                # pick those transforms randomly
                aug_transform = transforms.Compose(
                    random.sample(
                        self.augmentation_list,
                        num_transforms,
                    )
                )
                aug_image = aug_transform(original_image)
                self.processed_data.append((aug_image, item["encoded_genus"]))

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
