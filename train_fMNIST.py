import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import ConvAngularPen, ConvBaseline
from plotting import plot_2d, plot_3d


def train_baseline(train_loader, embed_dim):
    model = ConvBaseline(embed_dim=embed_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_step = len(train_loader)
    for epoch in tqdm(range(args.num_epochs), desc="Epochs"):
        for i, (feats, labels) in enumerate(tqdm(train_loader, leave=False, desc="Batches")):
            feats = feats.to(device)
            labels = labels.to(device)
            out = model(feats)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                tqdm.write(
                    "Baseline: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, args.num_epochs, i + 1, total_step, loss.item()
                    )
                )
        if (epoch + 1) % 8 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] / 4
    return model.cpu()


def train_am(train_loader, embed_dim, loss_type):
    model = ConvAngularPen(embed_dim=embed_dim, loss_type=loss_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_step = len(train_loader)
    for epoch in tqdm(range(args.num_epochs), desc=f"{loss_type} Epochs"):
        for i, (feats, labels) in enumerate(tqdm(train_loader, leave=False, desc="Batches")):
            feats = feats.to(device)
            labels = labels.to(device)
            loss = model(feats, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                tqdm.write(
                    "{}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        loss_type, epoch + 1, args.num_epochs, i + 1, total_step, loss.item()
                    )
                )

        if (epoch + 1) % 8 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] / 4

    return model.cpu()


def get_embeds(model, loader):
    model = model.to(device).eval()
    full_embeds = []
    full_labels = []
    with torch.no_grad():
        for i, (feats, labels) in enumerate(loader):
            feats = feats[:100].to(device)
            full_labels.append(labels[:100].cpu().detach().numpy())
            embeds = model(feats, embed=True)
            # NOTE: Here the embeddings are normalized to lie on the unit sphere
            full_embeds.append(F.normalize(embeds.detach().cpu()).numpy())
    model = model.cpu()
    return np.concatenate(full_embeds), np.concatenate(full_labels)


def main(args: argparse.Namespace):
    train_ds = datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
        download=True,
    )
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    example_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=False)

    os.makedirs("./figs_custom", exist_ok=True)

    print("Training Baseline model....")
    model_baseline = train_baseline(train_loader, args.embed_dim)
    bl_embeds, bl_labels = get_embeds(model_baseline, example_loader)
    if args.embed_dim == 2:
        plot_2d(bl_embeds, bl_labels, fig_path="./figs_custom/baseline.png")
    else:
        plot_3d(bl_embeds, bl_labels, fig_path="./figs_custom/baseline.png")
    print("Saved Baseline figure")

    del model_baseline, bl_embeds, bl_labels

    loss_types = ["arcface", "cosface", "sphereface"]
    for loss_type in loss_types:
        print(f"Training {loss_type} model....")
        model_am = train_am(train_loader, args.embed_dim, loss_type)
        am_embeds, am_labels = get_embeds(model_am, example_loader)
        if args.embed_dim == 2:
            plot_2d(am_embeds, am_labels, fig_path=f"./figs_custom/{loss_type}.png")
        else:
            plot_3d(am_embeds, am_labels, fig_path=f"./figs_custom/{loss_type}.png")
        print(f"Saved {loss_type} figure")
        del model_am, am_embeds, am_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Run Angular Penalty and Baseline experiments in fMNIST")
    parser.add_argument("--batch-size", type=int, default=512, help="input batch size for training (default: 512)")
    parser.add_argument("--embed_dim", type=int, default=3, help="Latent space dimension")
    parser.add_argument(
        "--num-epochs", type=int, default=40, help="Number of epochs to train each model for (default: 20)"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--use-cuda", action="store_true", default=False, help="enables CUDA training")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    main(args)
