import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import time


class MushroomTrainer:
    def __init__(
        self, model, train_dataloader, val_dataloader, test_dataloader, num_epochs
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.num_epochs = num_epochs

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.learning_rate_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)

    def train_model(self):
        epoch_train_losses = []
        epoch_val_losses = []
        for epoch in range(self.num_epochs):
            epoch_start = time.perf_counter()

            # train
            train_start = time.perf_counter()
            train_loss = self.train_epoch()
            train_time = time.perf_counter() - train_start
            epoch_train_losses.append(train_loss)

            # validate
            val_start = time.perf_counter()
            val_loss = self.run_model(self.val_dataloader)
            val_time = time.perf_counter() - val_start
            epoch_val_losses.append(val_loss)

            # print results
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"Epoch Train Loss: {train_loss:.4f} (took {train_time:.2f}s)")
            print(f"Epoch Val Loss: {val_loss:.4f} (took {val_time:.2f}s)")

            scheduler_start = time.perf_counter()
            self.learning_rate_scheduler.step()
            scheduler_time = time.perf_counter() - scheduler_start

            save_start = time.perf_counter()
            self.save_model(epoch)
            save_time = time.perf_counter() - save_start

            epoch_time = time.perf_counter() - epoch_start
            print(
                f"Epoch completed in {epoch_time:.2f}s (Scheduler: {scheduler_time:.2f}s, Save: {save_time:.2f}s)"
            )
            print("-" * 60)

        return epoch_train_losses, epoch_val_losses

    def train_epoch(self):
        epoch_train_loss = 0
        total_batch_time = 0
        forward_time = 0
        backward_time = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            batch_start = time.perf_counter()
            batch_train_loss, forward_time, backward_time = self.train_batch(
                inputs, targets
            )
            batch_time = time.perf_counter() - batch_start
            total_batch_time += batch_time

            epoch_train_loss += batch_train_loss
            if batch_idx % 50 == 0:
                print(
                    f"Batch {batch_idx}: Loss={batch_train_loss:.4f}, Time={batch_time:.3f}s"
                )
                print(f"  Forward: {forward_time:.3f}s, Backward: {backward_time:.3f}s")

        avg_epoch_train_loss = epoch_train_loss / len(self.train_dataloader)
        print(f"Average batch time: {total_batch_time/len(self.train_dataloader):.3f}s")
        return avg_epoch_train_loss

    def train_batch(self, inputs, targets):
        self.optimizer.zero_grad()

        forward_start = time.perf_counter()
        preds = self.model.forward(inputs)
        loss = self.loss_fn(preds, targets)
        forward_time = time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        loss.backward()
        self.optimizer.step()
        backward_time = time.perf_counter() - backward_start

        return loss.item(), forward_time, backward_time

    def run_model(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                preds = self.model.forward(inputs)
                loss = self.loss_fn(preds, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        self.model.train()
        return avg_loss

    def save_model(self, epoch):
        base_path = "models/"
        os.makedirs(base_path, exist_ok=True)
        i = 0
        while os.path.exists(os.path.join(base_path, f"model_{i}_epoch_{epoch}.pth")):
            i += 1

        model_path = os.path.join(base_path, f"model_{i}_epoch_{epoch}.pth")
        trainer_path = os.path.join(base_path, f"model_trainer_{i}_epoch_{epoch}.pth")
        print(f"saving model at {model_path}")
        torch.save(self.model, model_path)
        torch.save(self, trainer_path)
