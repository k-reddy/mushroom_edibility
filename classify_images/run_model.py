import kagglehub
from torch.utils.data import DataLoader
import random
from utils import load_images, encode_labels, split_data
from mushroom_dataset import MushroomDataset
from cnn import MushroomClassifier
from trainer import MushroomTrainer
import time


def create_data_lists(base_dir):
    print("creating data lists")
    images = load_images(base_dir=base_dir)
    images = encode_labels(images=images)
    random.shuffle(images)

    # returns train data, val data, test data
    return split_data(images)


def make_dataloader(data_list, num_augmentations=0):
    start = time.perf_counter()
    shroom_dataset = MushroomDataset(data_list, num_augmentations=num_augmentations)
    print(f"Dataset creation time: {time.perf_counter()-start}")
    return DataLoader(shroom_dataset, batch_size=32, shuffle=True, num_workers=3)


def main():
    # BASE_DIR = kagglehub.dataset_download("maysee/mushrooms-classification-common-genuss-images")
    # print("Path to dataset files:", path)
    BASE_DIR = "/Users/keerthireddy/.cache/kagglehub/datasets/maysee/mushrooms-classification-common-genuss-images/versions/1/Mushrooms"
    train_data, val_data, test_data = create_data_lists(BASE_DIR)

    # make dataloaders
    train_dataloader = make_dataloader(train_data, num_augmentations=1)
    val_dataloader = make_dataloader(val_data)
    test_dataloader = make_dataloader(test_data)

    num_classes = len({data["genus"] for data in train_data})

    shroom_classifier = MushroomClassifier(num_classes)
    shroom_trainer = MushroomTrainer(
        shroom_classifier,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        num_epochs=5,
    )

    print("training model")
    shroom_trainer.train_model()


if __name__ == "__main__":
    main()
