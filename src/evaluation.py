import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scipy.linalg import sqrtm

from generator import Generator
from data_loader import get_dataloader



LATENT_DIM = 100
NUM_FAKE_IMAGES = 1000
BATCH_SIZE = 64
CHECKPOINT_NAME = "G_epoch_100.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", CHECKPOINT_NAME)



class FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128)
        )

    def forward(self, x):
        return self.net(x)



def load_generator():
    G = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    G.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    G.eval()
    return G


def generate_fake_images(G, num_images):
    images = []
    with torch.no_grad():
        for _ in range(num_images // BATCH_SIZE):
            z = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
            fake = G(z)
            images.append(fake.cpu())
    return torch.cat(images, dim=0)


def extract_features(model, images):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i+BATCH_SIZE].to(DEVICE)
            emb = model(batch)
            features.append(emb.cpu())
    return torch.cat(features, dim=0).numpy()


def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)



def evaluate():
    print("Loading real dataset...")
    dataloader = get_dataloader(batch_size=BATCH_SIZE)

    real_images = []
    for imgs, _ in dataloader:
        real_images.append(imgs)
        if len(torch.cat(real_images)) >= NUM_FAKE_IMAGES:
            break
    real_images = torch.cat(real_images)[:NUM_FAKE_IMAGES]

    print("Loading generator...")
    G = load_generator()

    print("Generating synthetic images...")
    fake_images = generate_fake_images(G, NUM_FAKE_IMAGES)

    print("Extracting features...")
    feature_net = FeatureExtractor().to(DEVICE)

    real_feats = extract_features(feature_net, real_images)
    fake_feats = extract_features(feature_net, fake_images)

    print("Computing FID proxy...")
    mu_real, sigma_real = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_fake, sigma_fake = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    fid_score = compute_fid(mu_real, sigma_real, mu_fake, sigma_fake)

    # Diversity score
    diversity = np.mean(np.std(fake_feats, axis=0))

    # Nearest-neighbor
    distances = np.linalg.norm(
        real_feats[:, None, :] - fake_feats[None, :, :],
        axis=2
    )
    min_distance = distances.min()

    print("\n--- Evaluation Results ---")
    print(f"FID Proxy Score       : {fid_score:.4f}")
    print(f"Diversity Score       : {diversity:.4f}")
    print(f"Min NN Distance (Priv): {min_distance:.4f}")


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    evaluate()
