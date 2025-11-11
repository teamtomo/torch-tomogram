import mrcfile
import torch
from pathlib import Path
from torch_tomogram import Tomogram


# Path to ETOMO project directory
ETOMO_DIR = Path("/path/to/etomo/dir")

# Choose device: "cpu" or "cuda" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional: Pixel spacing in Angstroms
# If provided, shifts from alignment will be stored in Angstroms
# If None, shifts remain in pixels 
PIXEL_SPACING = 6.192

# Load tilt series
tilt_series = Tomogram.from_etomo_directory(
    etomo_dir=ETOMO_DIR,
    pixel_spacing=PIXEL_SPACING,
    device=DEVICE,
)


# Reconstruct tomogram
volume_shape = (512, 512, 512)
sidelength = 128
tomogram = tilt_series.reconstruct_tomogram(volume_shape, sidelength)

# Save as MRC file
output_path = ETOMO_DIR / 'torch_tomogram_reconstruction.mrc'
mrcfile.write(
    output_path,
    tomogram.cpu().numpy(),
    overwrite=True,
    voxel_size=10,
)

