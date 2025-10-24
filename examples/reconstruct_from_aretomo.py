import alnfile
import mrcfile
import numpy as np
from pathlib import Path
from torch_tomogram import Tomogram

# Paths
ALN_PATH = Path('/Path/to/your/alnfile.aln')
TILT_STACK_PATH = Path('/Path/to/your/raw/tilt_stack.ts_ext')

# Read AreTomo alignment data
df = alnfile.read(ALN_PATH)

# Get shifts for reconstruction 
# In aretomo tx and ty already represents the shifts in the forward projection model (sample -> image) 
# so we can either use directly those
corrected_shifts_t_xy = df[['tx', 'ty']].to_numpy()
corrected_shifts_t_yx = corrected_shifts_t_xy[:, ::-1].copy() #yx

# or get IMOD xf components from the dataframe
# df_to_xf(df, yx=True) returns (n_tilts, 2, 3) array
# Each matrix is [[A22, A21, DY], [A12, A11, DX]] (ready for torch-tomogram yz) 
xf = alnfile.df_to_xf(df, yx=True)
m, shifts = xf[:, :, :2], xf[:, :, 2]
# Convert IMOD's backward projection model to torch-tomogram's forward model
# IMOD: image -> sample
#   > the 2d matrix from the .xf file represents a 2d transform to align
#   > the image with the tilt-axis
# torch-tomogram: sample -> image
#   > the shifts are applied after rotation and projection and shift the
#   > projected sample to the image position
#
#  Roation matrix are orthogonal, so inversion = transposition :
#  np.einsum('nij,nj->ni', np.linalg.inv(m), shifts) = np.einsum('nji,nj->ni', m, shifts) 
#
#  Negate shifts for forward projection model
corrected_shifts_xf = -np.einsum('nji,nj->ni', m, shifts)

# Load tilt stack
tilt_stack_full = mrcfile.read(TILT_STACK_PATH)

included_indices = df['sec'].values - 1  # 0-indexed section indices
tilt_stack = tilt_stack_full[included_indices]

# Normalize
tilt_stack -= np.mean(tilt_stack, axis=(-2, -1), keepdims=True)
tilt_stack /= np.std(tilt_stack, axis=(-2, -1), keepdims=True) 

# Build tomogram and reconstruct
tilt_series = Tomogram(
    images=tilt_stack,
    tilt_angles=df['tilt'].to_numpy(),
    tilt_axis_angle=df['rot'].to_numpy(),  # Use single tilt axis angle
    sample_translations=corrected_shifts_xf.copy() # or corrected_shifts_t_yx.copy()
) 

tomogram = tilt_series.reconstruct_tomogram((100, 480, 320), 128)

# Save tomogram
out_path = ALN_PATH.parent / 'tt_rec.mrc'
mrcfile.write(out_path, tomogram.numpy(), overwrite=True, voxel_size=10)
