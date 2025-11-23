import alnfile
import mrcfile
import torch
from pathlib import Path
from torch_tomogram import Tomogram


# Paths to AreTomo alignment file and raw tilt stack
ALN_PATH = Path("/path/to/your/alignment.aln")
TILT_STACK_PATH = Path("/path/to/your/tilt_stack.mrc")

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
# AreTomo .aln files contain shifts in pixels, this converts them to Angstroms
PIXEL_SPACING = 6.192

# Read AreTomo alignment data and add tilt stack path to dataframe
df = alnfile.read(ALN_PATH)
df['image_path'] = str(TILT_STACK_PATH)

# Load tilt series from dataframe
tilt_series = Tomogram.from_aretomo_output(
    df=df,
    pixel_spacing=PIXEL_SPACING,
    device=DEVICE,
)

# Reconstruct tomogram
volume_shape = (512, 512, 512)
sidelength = 128
tomogram = tilt_series.reconstruct_tomogram(volume_shape, sidelength)

# Save as MRC file
output_path = ALN_PATH.parent / 'torch_tomogram_reconstruction.mrc'
mrcfile.write(
    output_path,
    tomogram.cpu().numpy(),
    overwrite=True,
    voxel_size=PIXEL_SPACING,
)
