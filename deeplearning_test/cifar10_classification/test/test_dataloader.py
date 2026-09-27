from datasets.cifar10_dataset import (
    get_train_dataloader,
    get_test_dataloader
)


train_dataloader = get_train_dataloader()
test_dataloader = get_test_dataloader()


train_images, train_labels = next(iter(train_dataloader))
test_images, test_labels = next(iter(test_dataloader))


print("===== Train =====")
print("images shape:", train_images.shape)
print("labels shape:", train_labels.shape)

print()

print("===== Test =====")
print("images shape:", test_images.shape)
print("labels shape:", test_labels.shape)