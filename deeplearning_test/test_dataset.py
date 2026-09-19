from mnist_dataset import MNISTDataset


dataset = MNISTDataset(
    "../data/MNIST/raw/train-images-idx3-ubyte",
    "../data/MNIST/raw/train-labels-idx1-ubyte"
)


print(len(dataset))


img,label = dataset[0]


print(img.shape)
print(label)