import os
import io
import zipfile
import torch
import streamlit as st
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from generator import Generator



LATENT_DIM = 100
CHECKPOINT_NAME = "G_epoch_100.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", CHECKPOINT_NAME)



@st.cache_resource
def load_generator():
    G = Generator(latent_dim=LATENT_DIM).to(DEVICE)
    G.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    G.eval()
    return G



st.set_page_config(page_title="Synthetic Chest X-ray Generator", layout="centered")

st.title("🫁 Chest X-ray Generator")
st.write(
    "Generate synthetic chest X-ray images using a trained Vanilla GAN"
)

# sidebar
st.sidebar.header("Generation Controls")

num_images = st.sidebar.slider(
    "Number of images",
    min_value=1,
    max_value=32,
    value=16,
    step=1
)

seed = st.sidebar.number_input(
    "Random seed (for reproducibility)",
    min_value=0,
    value=42,
    step=1
)

generate_button = st.sidebar.button("Generate Images")


# generator logic
if generate_button:
    st.write("Generating synthetic images...")

    torch.manual_seed(seed)

    G = load_generator()
    z = torch.randn(num_images, LATENT_DIM, device=DEVICE)

    with torch.no_grad():
        fake_images = G(z).cpu()

    # Make image grid
    grid = make_grid(fake_images, nrow=4, normalize=True)
    pil_img = to_pil_image(grid)

    st.image(pil_img, caption="Synthetic Chest X-ray Samples")


    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, img in enumerate(fake_images):
            img_pil = to_pil_image((img + 1) / 2)  # [-1,1] → [0,1]
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            zip_file.writestr(f"synthetic_xray_{i+1}.png", img_bytes.getvalue())

    st.download_button(
        label="Download images as ZIP",
        data=zip_buffer.getvalue(),
        file_name="synthetic_chest_xrays.zip",
        mime="application/zip"
    )
