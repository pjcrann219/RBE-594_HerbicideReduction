from Utilities import *

train_dir = "ComputerVision/Agriculture-Vision-2021/train"
train_loader = get_dataloader(train_dir)

test_dir = "ComputerVision/Agriculture-Vision-2021/val"
test_loader = get_dataloader(test_dir)


for input, labels, in train_loader:
    print(input.shape)
    print(labels.shape)
    break