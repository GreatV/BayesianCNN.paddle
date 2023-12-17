import paddle
import numpy as np


class CustomDataset(paddle.io.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def extract_classes(dataset, classes):
    idx = paddle.zeros_like(x=dataset.targets, dtype="bool")
    for target in classes:
        idx = idx | (dataset.targets == target)
    data, targets = dataset.data[idx], dataset.targets[idx]
    return data, targets


def getDataset(dataset):
    transform_split_mnist = paddle.vision.transforms.Compose(
        [
            paddle.vision.transforms.ToPILImage(),
            paddle.vision.transforms.Resize((32, 32)),
            paddle.vision.transforms.ToTensor(),
        ]
    )
    transform_mnist = paddle.vision.transforms.Compose(
        [paddle.vision.transforms.Resize((32, 32)), paddle.vision.transforms.ToTensor()]
    )
    transform_cifar = paddle.vision.transforms.Compose(
        [
            paddle.vision.transforms.Resize((32, 32)),
            paddle.vision.transforms.RandomHorizontalFlip(),
            paddle.vision.transforms.ToTensor(),
        ]
    )
    if dataset == "CIFAR10":
        trainset = paddle.vision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_cifar
        )
        testset = paddle.vision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_cifar
        )
        num_classes = 10
        inputs = 3
    elif dataset == "CIFAR100":
        trainset = paddle.vision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_cifar
        )
        testset = paddle.vision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_cifar
        )
        num_classes = 100
        inputs = 3
    elif dataset == "MNIST":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        num_classes = 10
        inputs = 1
    elif dataset == "SplitMNIST-2.1":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        train_data, train_targets = extract_classes(trainset, [0, 1, 2, 3, 4])
        test_data, test_targets = extract_classes(testset, [0, 1, 2, 3, 4])
        trainset = CustomDataset(
            train_data, train_targets, transform=transform_split_mnist
        )
        testset = CustomDataset(
            test_data, test_targets, transform=transform_split_mnist
        )
        num_classes = 5
        inputs = 1
    elif dataset == "SplitMNIST-2.2":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        train_data, train_targets = extract_classes(trainset, [5, 6, 7, 8, 9])
        test_data, test_targets = extract_classes(testset, [5, 6, 7, 8, 9])
        train_targets -= 5
        test_targets -= 5
        trainset = CustomDataset(
            train_data, train_targets, transform=transform_split_mnist
        )
        testset = CustomDataset(
            test_data, test_targets, transform=transform_split_mnist
        )
        num_classes = 5
        inputs = 1
    elif dataset == "SplitMNIST-5.1":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        train_data, train_targets = extract_classes(trainset, [0, 1])
        test_data, test_targets = extract_classes(testset, [0, 1])
        trainset = CustomDataset(
            train_data, train_targets, transform=transform_split_mnist
        )
        testset = CustomDataset(
            test_data, test_targets, transform=transform_split_mnist
        )
        num_classes = 2
        inputs = 1
    elif dataset == "SplitMNIST-5.2":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        train_data, train_targets = extract_classes(trainset, [2, 3])
        test_data, test_targets = extract_classes(testset, [2, 3])
        train_targets -= 2
        test_targets -= 2
        trainset = CustomDataset(
            train_data, train_targets, transform=transform_split_mnist
        )
        testset = CustomDataset(
            test_data, test_targets, transform=transform_split_mnist
        )
        num_classes = 2
        inputs = 1
    elif dataset == "SplitMNIST-5.3":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        train_data, train_targets = extract_classes(trainset, [4, 5])
        test_data, test_targets = extract_classes(testset, [4, 5])
        train_targets -= 4
        test_targets -= 4
        trainset = CustomDataset(
            train_data, train_targets, transform=transform_split_mnist
        )
        testset = CustomDataset(
            test_data, test_targets, transform=transform_split_mnist
        )
        num_classes = 2
        inputs = 1
    elif dataset == "SplitMNIST-5.4":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        train_data, train_targets = extract_classes(trainset, [6, 7])
        test_data, test_targets = extract_classes(testset, [6, 7])
        train_targets -= 6
        test_targets -= 6
        trainset = CustomDataset(
            train_data, train_targets, transform=transform_split_mnist
        )
        testset = CustomDataset(
            test_data, test_targets, transform=transform_split_mnist
        )
        num_classes = 2
        inputs = 1
    elif dataset == "SplitMNIST-5.5":
        trainset = paddle.vision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_mnist
        )
        testset = paddle.vision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_mnist
        )
        train_data, train_targets = extract_classes(trainset, [8, 9])
        test_data, test_targets = extract_classes(testset, [8, 9])
        train_targets -= 8
        test_targets -= 8
        trainset = CustomDataset(
            train_data, train_targets, transform=transform_split_mnist
        )
        testset = CustomDataset(
            test_data, test_targets, transform=transform_split_mnist
        )
        num_classes = 2
        inputs = 1
    return trainset, testset, inputs, num_classes


def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = paddle.io.SubsetRandomSampler(train_idx)
    valid_sampler = paddle.io.SubsetRandomSampler(valid_idx)
    train_loader = paddle.io.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = paddle.io.DataLoader(
        trainset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )
    test_loader = paddle.io.DataLoader(
        dataset=testset, batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, valid_loader, test_loader
