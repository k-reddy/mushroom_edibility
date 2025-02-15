import kagglehub
from torch.utils.data import DataLoader
import random
from utils import load_images, encode_labels, split_data
from mushroom_dataset import MushroomDataset
from cnn import MushroomClassifier
from trainer import MushroomTrainer
import time

# base_dir = kagglehub.dataset_download("maysee/mushrooms-classification-common-genuss-images")
# print("Path to dataset files:", path)
base_dir = "/Users/keerthireddy/.cache/kagglehub/datasets/maysee/mushrooms-classification-common-genuss-images/versions/1/Mushrooms"
print("loading images")
images = load_images(base_dir=base_dir)
images = encode_labels(images=images)
random.shuffle(images)

print("splitting data")
train_data, val_data, test_data = split_data(images)
train_start_time = time.perf_counter()
print("creating datasets")

# make datasets and dataloaders
train_dataset = MushroomDataset(train_data, num_augmentations=1)
train_time = time.perf_counter()
print(f"Train dataset creation time: {train_time-train_start_time:.2f} seconds")

val_dataset = MushroomDataset(val_data, num_augmentations=0)
val_time = time.perf_counter()
print(f"Validation dataset creation time: {val_time-train_time:.2f} seconds")

test_dataset = MushroomDataset(test_data, num_augmentations=0)
test_time = time.perf_counter()
print(f"Test dataset creation time: {test_time-val_time:.2f} seconds")

print("creating dataloaders")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=3)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=3)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=3)

num_classes = len(set([data["genus"] for data in train_data]))
shroom_classifier = MushroomClassifier(num_classes)
shroom_trainer = MushroomTrainer(
    shroom_classifier, train_dataloader, val_dataloader, test_dataloader, num_epochs=5
)

print("training model")
shroom_trainer.train_model()
