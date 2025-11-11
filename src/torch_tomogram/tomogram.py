"""Tomogram reconstruction in pytorch."""

from pathlib import Path

import etomofiles
import alnfile
import mrcfile
import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Ry, Rz, T
from torch_fourier_slice import insert_central_slices_rfft_3d_multichannel
from torch_grid_utils import dft_center, fftfreq_grid
from torch_subpixel_crop import subpixel_crop_2d


class Tomogram:
    """Tomogram class that enables reconstruction position querying."""

    def __init__(
        self,
        tilt_angles: torch.Tensor,
        tilt_axis_angle: torch.Tensor,
        sample_translations: torch.Tensor,
        images: torch.Tensor,  # (b, h, w)
        pixel_spacing: float | None = None,
        device: torch.device | str = "cpu",
    ):
        self.images = torch.as_tensor(images, device=device).float()
        self.tilt_angles = torch.as_tensor(tilt_angles, device=device).float()
        self.tilt_axis_angle = torch.as_tensor(tilt_axis_angle, device=device).float()
        self.sample_translations = torch.as_tensor(
            sample_translations, device=device
        ).float()
        self.pixel_spacing = pixel_spacing
        self.device = device
        self._pad_factor = 2.0
    
    @property
    def sample_translations_px(self) -> torch.Tensor:
        """Get sample translations in pixels.
        
        If pixel_spacing was provided, converts from Angstroms to pixels.
        Otherwise, returns the translations as-is (already in pixels).
        """
        if self.pixel_spacing is not None:
            return self.sample_translations / self.pixel_spacing
        return self.sample_translations

    @classmethod
    def from_aretomo_aln(
        cls,
        aln_path: Path | str,
        tilt_stack_path: Path | str,
        pixel_spacing: float | None = None,
        device: torch.device | str = "cpu",
    ) -> "Tomogram":
        """Initialize Tomogram from AreTomo alignment file and tilt stack."""

        aln_path = Path(aln_path)
        tilt_stack_path = Path(tilt_stack_path)
        df = alnfile.read(aln_path)
        corrected_shifts_xy = df[["tx", "ty"]].to_numpy()
        corrected_shifts_yx = corrected_shifts_xy[:, ::-1].copy()
        
        # Convert shifts to Angstroms if pixel_spacing provided
        if pixel_spacing is not None:
            corrected_shifts_yx = corrected_shifts_yx * pixel_spacing
        
        tilt_stack_full = mrcfile.read(tilt_stack_path)
        included_indices = df["sec"].values - 1 
        tilt_stack = tilt_stack_full[included_indices]
        tilt_stack = tilt_stack.astype(np.float32)
        tilt_stack -= np.mean(tilt_stack, axis=(-2, -1), keepdims=True)
        tilt_stack /= np.std(tilt_stack, axis=(-2, -1), keepdims=True)
        return cls(
            images=tilt_stack,
            tilt_angles=df["tilt"].to_numpy(),
            tilt_axis_angle=df["rot"].to_numpy(),
            sample_translations=corrected_shifts_yx,
            pixel_spacing=pixel_spacing,
            device=device,
        )

    @classmethod
    def from_etomo_directory(
        cls,
        etomo_dir: Path | str,
        pixel_spacing: float | None = None,
        device: torch.device | str = "cpu",
    ) -> "Tomogram":
        """Initialize Tomogram from ETOMO directory."""

        etomo_dir = Path(etomo_dir)
        df = etomofiles.read(etomo_dir)
        df = df.loc[~df["excluded"]].reset_index(drop=True)
        xf = etomofiles.df_to_xf(df, yx=True)
        m, shifts = xf[:, :, :2], xf[:, :, 2]
        corrected_shifts = -np.einsum("nji,nj->ni", m, shifts)
        corrected_shifts = np.ascontiguousarray(corrected_shifts)
        # Convert shifts to Angstroms if pixel_spacing is available
        if pixel_spacing is not None:
            corrected_shifts = corrected_shifts * pixel_spacing
        tilt_stack_path = etomo_dir / df.image_path[0].replace("[0]", "")
        tilt_stack_full = mrcfile.read(tilt_stack_path)
        tilt_stack = tilt_stack_full[df.idx_tilt.to_numpy()]
        tilt_stack = tilt_stack.astype(np.float32)
        tilt_stack -= np.mean(tilt_stack, axis=(-2, -1), keepdims=True)
        tilt_stack /= np.std(tilt_stack, axis=(-2, -1), keepdims=True)

        return cls(
            images=tilt_stack,
            tilt_angles=df.tlt.to_numpy(),
            tilt_axis_angle=df.tilt_axis_angle.to_numpy(),
            sample_translations=corrected_shifts,
            pixel_spacing=pixel_spacing,
            device=device,
        )

    @property
    def projection_matrices(self) -> torch.Tensor:
        """Matrices that project points from 3D -> 2D."""
        shifts_3d = F.pad(self.sample_translations_px, (1, 0), value=0)
        r0 = Ry(self.tilt_angles, zyx=True, device=self.device)
        r1 = Rz(self.tilt_axis_angle, zyx=True, device=self.device)
        t2 = T(shifts_3d, device=self.device)
        return t2 @ r1 @ r0

    def to(self, device: torch.device | str) -> None:
        """Move all objects of the tomogram to the device."""
        self.device = device
        self.images = self.images.to(device)
        self.tilt_angles = self.tilt_angles.to(device)
        self.tilt_axis_angle = self.tilt_axis_angle.to(device)
        self.sample_translations = self.sample_translations.to(device)

    def project_points(self, points_zyx: torch.Tensor) -> torch.Tensor:
        """Project points from 3D to 2D.

        - points are 3D zyx coordinates
        - points are positions relative to center of tomogram
        - projected 2D points are relative to center of 2D image
        """
        points_zyx = torch.as_tensor(points_zyx, device=self.device).float()
        M_yx = self.projection_matrices[..., [1, 2], :]  # (ntilts, 2, 4)
        points_zyxw = homogenise_coordinates(points_zyx)
        projected_yx = M_yx @ einops.rearrange(
            points_zyxw, "nparticles zyxw -> nparticles 1 zyxw 1"
        )
        projected_yx = einops.rearrange(
            projected_yx, "nparticles ntilts yx 1 -> nparticles ntilts yx"
        )
        return projected_yx  # (points, tilts, yx)

    def extract_particle_tilt_series(
        self, points_zyx: torch.Tensor, sidelength: int, return_rfft: bool = True
    ) -> torch.Tensor:
        """Extract a subtilt-series at a 3D location in the sample."""
        projected_yx = self.project_points(points_zyx)
        projected_yx += dft_center(
            self.images.shape[-2:], rfft=False, fftshift=True, device=self.device
        )
        images = subpixel_crop_2d(
            image=self.images,
            positions=projected_yx,
            sidelength=sidelength,
            return_rfft=return_rfft,
            decenter=return_rfft,  
        )
        return images

    def reconstruct_subvolume(
        self, point_zyx: torch.Tensor, sidelength: int
    ) -> torch.Tensor:
        """Reconstruct a 3D patch at a location in the sample."""
        # Use batched method and extract first result
        patches = self.reconstruct_patches_batched(point_zyx, sidelength)
        return patches[0]

    def reconstruct_tomogram(
        self, volume_shape: tuple[int, int, int], sidelength: int
    ) -> torch.Tensor:
        """Reconstruct the full tomogram by tiling reconstructed patches in 3D. """
        d, h, w = volume_shape
        r = sidelength // 2

        # Setup grid points where patches will be reconstructed
        z = torch.arange(start=r, end=d + r, step=sidelength, device=self.device) - d // 2
        y = torch.arange(start=r, end=h + r, step=sidelength, device=self.device) - h // 2
        x = torch.arange(start=r, end=w + r, step=sidelength, device=self.device) - w // 2

        # Create grid of all positions: (n_z, n_y, n_x, 3)
        grid_zyx = torch.stack(torch.meshgrid(z, y, x, indexing='ij'), dim=-1)
        original_shape = grid_zyx.shape[:-1]  # (n_z, n_y, n_x)
        grid_zyx = grid_zyx.reshape(-1, 3)  # (N, 3) where N = n_z * n_y * n_x
        
        # Reconstruct all patches at once
        patches = self.reconstruct_patches_batched(
            points_zyx=grid_zyx, 
            sidelength=sidelength
        )  # (N, sidelength, sidelength, sidelength)
        
        # Reshape back to grid structure: (n_z, n_y, n_x, sidelength, sidelength, sidelength)
        patches = patches.reshape(*original_shape, sidelength, sidelength, sidelength)
        
        # Tile all patches into the full volume
        tomogram = einops.rearrange(
            patches,
            'nz ny nx d h w -> (nz d) (ny h) (nx w)'
        )
        
        # Crop to desired volume shape 
        tomogram = tomogram[:d, :h, :w]
        
        return tomogram
    
    def reconstruct_patches_batched(
        self, points_zyx: torch.Tensor, sidelength: int
    ) -> torch.Tensor:
        """Reconstruct multiple 3D patches simultaneously from 2D projections. """
        points_zyx = torch.as_tensor(points_zyx, device=self.device).float()
        if points_zyx.ndim == 1:
            points_zyx = points_zyx.reshape(1, 3)
        
        n_positions = points_zyx.shape[0]
        rotation_matrices = self.projection_matrices[:, :3, :3]
        rotation_matrices = torch.linalg.pinv(rotation_matrices)
        sidelength_padded = int(self._pad_factor * sidelength)
        
        # Extract patches (batched)
        particle_tilt_series_rfft = self.extract_particle_tilt_series(
            points_zyx, sidelength=sidelength_padded, return_rfft=True
        )  # (n_positions, n_tilts, h, w_rfft)
        
        particle_tilt_series_rfft = torch.fft.fftshift(
            particle_tilt_series_rfft, dim=(-2,)
        )  # (n_positions, n_tilts, h, w_rfft)
        
        # Transpose to multichannel format: (n_tilts, n_positions, h, w_rfft)
        # where n_tilts is the batch dim and n_positions is the channel dim
        particle_tilt_series_rfft = particle_tilt_series_rfft.transpose(0, 1)
        
        # Reconstruct all patches at once using multichannel insertion
        # Treat each patch as a separate "channel"
        patches_rfft, weights = insert_central_slices_rfft_3d_multichannel(
            image_rfft=particle_tilt_series_rfft,
            volume_shape=(sidelength_padded, sidelength_padded, sidelength_padded),
            rotation_matrices=rotation_matrices,
            zyx_matrices=True,
            fftfreq_max=0.5,
        )  # patches_rfft: (n_positions, d, h, w_rfft), weights: (d, h, w_rfft)
        
        # Reweight each patch
        valid_weights = weights > 1e-3
        patches_rfft[:, valid_weights] /= weights[valid_weights]
        
        patches_rfft = torch.fft.ifftshift(patches_rfft, dim=(-3, -2))

        patches = torch.fft.irfftn(
            patches_rfft, 
            s=(sidelength_padded,) * 3, 
            dim=(-3, -2, -1)
        )  # (n_positions, d, h, w)
        
        # Center all patches in real space
        patches = torch.fft.ifftshift(patches, dim=(-3, -2, -1))
        
        # Correct for convolution with linear interpolation kernel
        grid = fftfreq_grid(
            image_shape=(sidelength_padded, sidelength_padded, sidelength_padded), 
            rfft=False, fftshift=True, norm=True, device=self.device
        )
        patches = patches / torch.sinc(grid) ** 2
        
        # Remove padding from all patches
        p = (sidelength_padded - sidelength) // 2
        patches = F.pad(patches, [-p] * 6)
        
        return patches
