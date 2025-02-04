import os
from torchvision import transforms, datasets
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch
import global_v as glv
from torch.utils.data import DataLoader
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image



def load_mvtec(data_path, batch_size=None, input_size=None, small=False):
    print("Loading MVTEC dataset...")

    # Ensure the data path exists
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Default values for batch_size and input_size
    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    #mean = [0.4914, 0.4822, 0.4465]
    #std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        SetRange
    ])


    # Paths for training and testing datasets
    current_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.dirname(current_dir) + "/data/MVTEC/"
    train_data_path = current_dir + "hazelnut/train/good"
    test_data_path = current_dir + "hazelnut/test/good"

    # Define custom dataset class for single-class data
    class SingleClassDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = 0  # Assign a single class label (e.g., "good")
            return image, label

    # Use custom dataset for training and testing
    trainset = SingleClassDataset(root_dir=train_data_path, transform=transform_train)
    testset = SingleClassDataset(root_dir=test_data_path, transform=transform_test)

    # Data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader



def load_mnist(data_path, batch_size=None, input_size=None, small=False):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader


def load_fashionmnist(data_path, batch_size=None, input_size=None, small=False):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader


def load_cifar10(data_path, batch_size=None, input_size=None, small=False):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader


def load_celebA(data_path, batch_size=None, input_size=None, small=False):
    print("loading CelebA")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if batch_size is None:
        batch_size = glv.network_config['batch_size']
    if input_size is None:
        input_size = glv.network_config['input_size']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    test_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        transforms.CenterCrop(148),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path,
                                           split='train',
                                           download=True,
                                           transform=transform)
    testset = torchvision.datasets.CelebA(root=data_path,
                                            split='test',
                                            download=True,
                                            transform=test_transform)

    if small:
        trainset, _ = random_split(dataset=trainset, lengths=[int(0.1 * len(trainset)), int(0.9 * len(trainset))],
                                   generator=torch.Generator().manual_seed(0))
        testset, _ = random_split(dataset=testset, lengths=[int(0.1 * len(testset)), int(0.9 * len(testset))],
                                   generator=torch.Generator().manual_seed(0))
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size*2,
                                            shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader



