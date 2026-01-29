import os
import torch
from torchvision.utils import save_image

from generator import Generator



LATENT_DIM = 100
NUM_IMAGES = 16
CHECKPOINT_NAME = "G_epoch_100.pth"   # change if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", CHECKPOINT_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, "samples", "inference")

os.makedirs(OUTPUT_DIR, exist_ok=True)



def load_generator(checkpoint_path):
    G = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    G.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    G.eval()
    return G



def generate_images(generator, num_images=16):
    z = torch.randn(num_images, LATENT_DIM, device=DEVICE)
    with torch.no_grad():
        fake_images = generator(z)
    return fake_images



if __name__ == "__main__":
    print("Loading generator from:", CHECKPOINT_PATH)

    G = load_generator(CHECKPOINT_PATH)
    images = generate_images(G, NUM_IMAGES)

    save_path = os.path.join(OUTPUT_DIR, "synthetic_samples.png")
    save_image(images, save_path, normalize=True, nrow=4)

    print(f"Saved {NUM_IMAGES} synthetic images to:")
    print(save_path)
