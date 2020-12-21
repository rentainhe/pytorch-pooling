from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def get_train_loader(configs):
    # data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(configs.img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(configs.mean, configs.std)
    ])
    trainset = datasets.CIFAR100(root='./data/cifar100',
                                train=True,
                                download=True,
                                transform=transform_train)
    # get dataloader
    train_sampler = RandomSampler(trainset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=configs.batch_size,
                              num_workers=configs.num_workers,
                              pin_memory=configs.pin_memory)
    return train_loader

def get_test_loader(configs):
    # data transforms
    transform_test = transforms.Compose([
        transforms.Resize((configs.img_size, configs.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(configs.mean, configs.std)
    ])
    testset = datasets.CIFAR100(root='./data/cifar100',
                                train=False,
                                download=True,
                                transform=transform_test)

    # get dataloader
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=configs.eval_batch_size,
                             num_workers=configs.num_workers,
                             pin_memory=configs.pin_memory)
    return test_loader
