# data/gtsrb.py
import torch
from torchvision import transforms
from data.gtsrb_poison import PoisonGTSRBDataset

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

def gtsrb_loader(path, batch_size=128, train=True):
    if path == "clean":
        path = None

    dataset = PoisonGTSRBDataset(
        root="datasets",
        train=train,
        transform=transform,
        poison_params=path
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=8
    )
    return dataset, loader
