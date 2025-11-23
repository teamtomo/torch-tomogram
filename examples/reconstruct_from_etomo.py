import etomofiles
import mrcfile
import torch
from pathlib import Path
from torch_tomogram import Tomogram


# Path to ETOMO project directory
ETOMO_DIR = Path("/path/to/etomo/dir")

# Choose device: "cpu" or "cuda" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
PIXEL_SPACING = 6.192

# Read ETOMO alignment data
df = etomofiles.read(ETOMO_DIR)

# Update image paths to be relative to current working directory (or absolute)
# etomofiles returns paths relative to the etomo directory
df['image_path'] = df['image_path'].apply(
    lambda p: str(ETOMO_DIR / p.split('[')[0]) + '[' + p.split('[')[1]
)

# Load tilt series from dataframe
tilt_series = Tomogram.from_etomo_directory(
    df=df,
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
    voxel_size=PIXEL_SPACING,
)

