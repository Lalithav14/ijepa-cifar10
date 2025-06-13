import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.ijepa import IJepaStudent
from utils import set_seed

def train(args):
    set_seed(42)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(args.train_root, transform=transform)
    val_dataset = datasets.ImageFolder(args.val_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = IJepaStudent()
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        model.train()
        print(f"[Epoch {epoch+1}] Starting training...")
        for images, _ in train_loader:
            images = images.to(args.device)

            loss = model(images, epoch)  # 🔥 epoch passed here
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")
# training logic here
