import os
import torch
from fastapi import FastAPI
from torchvision.utils import save_image

from generator import Generator


LATENT_DIM = 100
CHECKPOINT_NAME = "G_epoch_100.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", CHECKPOINT_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, "samples", "api")

os.makedirs(OUTPUT_DIR, exist_ok=True)


G = Generator(latent_dim=LATENT_DIM).to(DEVICE)
G.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
G.eval()


app = FastAPI(title="Synthetic Chest X-ray API")


@app.get("/")
def root():
    return {"message": "Synthetic Chest X-ray Generator API is running"}


@app.get("/generate")
def generate(num_images: int = 4):
    z = torch.randn(num_images, LATENT_DIM, device=DEVICE)

    with torch.no_grad():
        images = G(z)

    file_path = os.path.join(OUTPUT_DIR, "generated.png")
    save_image(images, file_path, normalize=True)

    return {
        "num_images": num_images,
        "saved_to": file_path
    }
