import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from src.generator import Generator
from src.discriminator import Discriminator
from src.data_loader import get_dataloader



EPOCHS = 100
BATCH_SIZE = 64
LATENT_DIM = 100
LR = 0.0002
BETA1 = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAMPLE_DIR = os.path.join(BASE_DIR, "samples")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Saving samples to:", SAMPLE_DIR)
print("Saving checkpoints to:", CHECKPOINT_DIR)


os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



dataloader = get_dataloader(batch_size=BATCH_SIZE)

G = Generator(latent_dim=LATENT_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))



for epoch in range(1, EPOCHS + 1):
    for i, (real_imgs, _) in enumerate(dataloader):

        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1, device=DEVICE)
        fake_labels = torch.zeros(batch_size, 1, device=DEVICE)


        optimizer_D.zero_grad()

        real_loss = criterion(D(real_imgs), real_labels)

        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()


        optimizer_G.zero_grad()

        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        gen_imgs = G(z)
        g_loss = criterion(D(gen_imgs), real_labels)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        # Logging
        # ---------------------
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{EPOCHS}] "
                f"[Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item():.4f}] "
                f"[G loss: {g_loss.item():.4f}]"
            )


    with torch.no_grad():
        sample_z = torch.randn(16, LATENT_DIM, device=DEVICE)
        samples = G(sample_z)
        save_image(samples, f"{SAMPLE_DIR}/epoch_{epoch:03d}.png", normalize=True)


    torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/G_epoch_{epoch:03d}.pth")
    torch.save(D.state_dict(), f"{CHECKPOINT_DIR}/D_epoch_{epoch:03d}.pth")