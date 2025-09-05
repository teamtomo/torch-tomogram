import etomofiles
import mrcfile
import numpy as np
from pathlib import Path
from torch_tomogram import Tomogram


# Read etomo alignment data
ETOMO_DIR = Path("/home/marten/data/datasets/apoferritin/TS_1/")
df = etomofiles.read(ETOMO_DIR)

# Convert IMOD's backward projection model to torch-tomogram's forward model
# IMOD: image -> sample
#   > the 2d matrix from the .xf file represents a 2d transform to align
#   > the image with the tilt-axis
# torch-tomogram: sample -> image
#   > the shifts are applied after rotation and projection and shift the
#   > projected sample to the image position

# Extract 2x2 transformation matrices from etomo data
m = df.loc[:, ['xf_a11', 'xf_a12', 'xf_a21', 'xf_a22']].to_numpy().reshape(-1, 2, 2)
# Invert matrices to convert from IMOD's backward to forward model
m = np.linalg.inv(m)

# Transform shifts from IMOD's coordinate system to torch-tomogram's
shifts = df.loc[:, ['xf_dx', 'xf_dy']].to_numpy()
# Apply inverse transformation to get post-projection shifts
corrected_shifts = np.einsum('nij,nj->ni', m, shifts)
# Negate shifts for forward projection model
corrected_shifts = corrected_shifts * -1

# Convert from IMOD's (x,y) to torch-tomogram's (y,x) convention
corrected_shifts = np.flip(corrected_shifts, axis=1)
corrected_shifts = np.ascontiguousarray(corrected_shifts)

# Load and normalize tilt stack
tilt_stack_path = ETOMO_DIR / df.image_path[0].replace('[0]', '')
tilt_stack = mrcfile.read(tilt_stack_path)
# Zero-mean, unit-variance normalization per image
tilt_stack -= np.mean(tilt_stack, axis=(-2, -1), keepdims=True)
tilt_stack /= np.std(tilt_stack, axis=(-2, -1), keepdims=True)

# Create tomogram object and reconstruct
tilt_series = Tomogram(
    images=tilt_stack,
    tilt_angles=df.tlt,
    tilt_axis_angle=df.tilt_axis_angle,
    sample_translations=corrected_shifts.copy()
)
tomogram = tilt_series.reconstruct_tomogram((100, 480, 342), 128)

mrcfile.write(ETOMO_DIR / 'tt_rec.mrc', tomogram.numpy(), overwrite=True, voxel_size=10)
