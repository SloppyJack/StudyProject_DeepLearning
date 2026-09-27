from datasets.cifar10_dataset import get_train_dataset


train_dataset = get_train_dataset()

print("训练集数量：", len(train_dataset))

image, label = train_dataset[0]

print("image 类型：", type(image))
print("label：", label)

print("image shape：", image.shape)
print("image：", image)