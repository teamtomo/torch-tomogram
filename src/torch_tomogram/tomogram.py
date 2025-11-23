"""Tomogram reconstruction in pytorch."""

from pathlib import Path

import etomofiles
import alnfile
import mrcfile
import einops
import numpy as np
import pandas as pd
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
        pixel_spacing: float,
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
        return self.sample_translations/self.pixel_spacing

    @classmethod
    def from_aretomo_output(
        cls,
        df: pd.DataFrame,
        pixel_spacing: float,
        device: torch.device | str = "cpu",
    ) -> "Tomogram":
        """Initialize Tomogram from AreTomo dataframe."""
        
        # Extract XY shifts and convert to YX convention
        corrected_shifts_xy = df[["tx", "ty"]].to_numpy()
        corrected_shifts_yx = corrected_shifts_xy[:, ::-1].copy()
        
        # Convert shifts from pixels to Angstroms
        corrected_shifts_yx_ang = corrected_shifts_yx * pixel_spacing
        
        # Load tilt stack and extract valid tilts
        tilt_stack_path = df['image_path'].iloc[0]
        tilt_stack_full = mrcfile.read(tilt_stack_path)
        idx_valid = df["sec"].values - 1  # Convert from 1-indexed to 0-indexed
        tilt_stack = tilt_stack_full[idx_valid]
        tilt_stack = tilt_stack.astype(np.float32)
        
        # Normalize on central 25% crop to avoid edge artifacts
        h, w = tilt_stack.shape[-2:]
        h_crop = slice(int(0.375 * h), int(0.625 * h))
        w_crop = slice(int(0.375 * w), int(0.625 * w))
        crop = tilt_stack[:, h_crop, w_crop]
        tilt_stack -= np.mean(crop, axis=(-2, -1), keepdims=True)
        tilt_stack /= np.std(crop, axis=(-2, -1), keepdims=True)
        return cls(
            images=tilt_stack,
            tilt_angles=df["tilt"].to_numpy(),
            tilt_axis_angle=df["rot"].to_numpy(),
            sample_translations=corrected_shifts_yx_ang,
            pixel_spacing=pixel_spacing,
            device=device,
        )

    @classmethod
    def from_etomo_directory(
        cls,
        df: pd.DataFrame,
        pixel_spacing: float,
        device: torch.device | str = "cpu",
    ) -> "Tomogram":
        """Initialize Tomogram from ETOMO directory."""
        df = df.loc[~df["excluded"]].reset_index(drop=True)
        # Get IMOD xf components from dataframe
        # df_to_xf(df, yx=True) returns (n_tilts, 2, 3) array
        # Each matrix is [[A22, A21, DY], [A12, A11, DX]] (ready for torch-tomogram yz) 
        xf = etomofiles.df_to_xf(df, yx=True)
        m, shifts = xf[:, :, :2], xf[:, :, 2]
        # Convert IMOD's backward projection model to torch-tomogram's forward model
        # IMOD: image -> sample
        #   > the 2d matrix from the .xf file represents a 2d transform to align
        #   > the images so that they represent projections of a solid body tilted around the Y axis
        # torch-tomogram: sample -> image
        #   > the shifts are applied after rotation and projection and shift the
        #   > projected sample to the image position
        #
        #  Roation matrix are orthogonal, so inversion = transposition :
        #  np.einsum('nij,nj->ni', np.linalg.inv(m), shifts) = np.einsum('nji,nj->ni', m, shifts) 
        #
        #  Negate shifts for forward projection model
        corrected_shifts = -np.einsum('nji,nj->ni', m, shifts)
        corrected_shifts = np.ascontiguousarray(corrected_shifts)
        
        # Convert shifts from pixels to Angstroms
        corrected_shifts_ang = corrected_shifts * pixel_spacing
        
        # Load tilt stack
        tilt_stack_path = df['image_path'][0].replace("[0]", "")
        tilt_stack_full = mrcfile.read(tilt_stack_path)
        tilt_stack = tilt_stack_full[df['idx_tilt'].to_numpy()]
        tilt_stack = tilt_stack.astype(np.float32)
        
        # Normalize on central 25% crop to avoid edge artifacts
        h, w = tilt_stack.shape[-2:]
        h_crop = slice(int(0.375 * h), int(0.625 * h))
        w_crop = slice(int(0.375 * w), int(0.625 * w))
        crop = tilt_stack[:, h_crop, w_crop]
        tilt_stack -= np.mean(crop, axis=(-2, -1), keepdims=True)
        tilt_stack /= np.std(crop, axis=(-2, -1), keepdims=True)

        return cls(
            images=tilt_stack,
            tilt_angles=df['tlt'].to_numpy(),
            tilt_axis_angle=df['tilt_axis_angle'].to_numpy(),
            sample_translations=corrected_shifts_ang,
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
        """Project 3D points to 2D image coordinates.

        - points are 3D zyx coordinates
        - points are positions relative to center of tomogram
        - projected 2D points are relative to center of 2D image
        """
        points_zyx = torch.as_tensor(points_zyx, device=self.device).float()
        
        # Convert from Angstroms to pixels for projection
        points_zyx_px = points_zyx / self.pixel_spacing
        
        # Apply projection matrices
        M_yx = self.projection_matrices[..., [1, 2], :]  # (ntilts, 2, 4)
        points_zyxw = homogenise_coordinates(points_zyx_px)
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
        self, points_zyx: torch.Tensor, sidelength: int
    ) -> torch.Tensor:
        """Reconstruct 3D patch(es) at location(s) in the sample.
        
        Rank-polymorphic: input (..., 3) -> output (..., d, h, w)
        """
        points_zyx = torch.as_tensor(points_zyx, device=self.device).float()
        
        points_zyx, ps = einops.pack([points_zyx], "* zyx")
        
        n_positions = points_zyx.shape[0]
        rotation_matrices = self.projection_matrices[:, :3, :3]
        rotation_matrices = torch.linalg.pinv(rotation_matrices)
        sidelength_padded = int(self._pad_factor * sidelength)
        
        # Extract patches (batched)
        particle_tilt_series_rfft = self.extract_particle_tilt_series(
            points_zyx, sidelength=sidelength_padded, return_rfft=True
        )  # (n_positions, n_tilts, h, w_rfft)
        
        # Apply fftshift on non-redundant dimension for central slice insertion
        particle_tilt_series_rfft = torch.fft.fftshift(
            particle_tilt_series_rfft, dim=(-2,)
        )  # (n_positions, n_tilts, h, w_rfft)
        
        # Rearrange to multichannel format: (n_tilts, n_positions, h, w_rfft)
        # where n_tilts is the batch dim and n_positions is the channel dim
        particle_tilt_series_rfft = einops.rearrange(
            particle_tilt_series_rfft, 
            "n_positions n_tilts h w_rfft -> n_tilts n_positions h w_rfft"
        )
        
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
        
        # Unpack to restore original input shape: (..., d, h, w)
        [patches] = einops.unpack(patches, ps, "* d h w")
        
        return patches

    def reconstruct_tomogram(
        self, volume_shape: tuple[int, int, int], sidelength: int, batch_size: int | None = None
    ) -> torch.Tensor:
        """Reconstruct the full tomogram by tiling reconstructed patches in 3D. """
        d, h, w = volume_shape
        r = sidelength // 2

        # Setup grid points where patches will be reconstructed (in pixels)
        z = torch.arange(start=r, end=d + r, step=sidelength, device=self.device) - d // 2
        y = torch.arange(start=r, end=h + r, step=sidelength, device=self.device) - h // 2
        x = torch.arange(start=r, end=w + r, step=sidelength, device=self.device) - w // 2

        # Create grid of all center points of reconstructed patches/subvolumes
        # Points are in pixels relative to volume center
        centers_zyx = torch.stack(torch.meshgrid(z, y, x, indexing='ij'), dim=-1)  # (gd, gh, gw, 3)
        
        # Convert positions from pixels to Angstroms
        centers_zyx_ang = centers_zyx * self.pixel_spacing
        
        # Reconstruct patches (optionally in batches)
        if batch_size is None:
            # Reconstruct all patches at once
            # Input: (gd, gh, gw, 3) -> Output: (gd, gh, gw, d, h, w)
            patches = self.reconstruct_subvolume(
                points_zyx=centers_zyx_ang, 
                sidelength=sidelength
            )
            # Tile all patches into the full volume
            tomogram = einops.rearrange(
                patches,
                'gd gh gw d h w -> (gd d) (gh h) (gw w)'
            )
        else:
            # reconstruct on GPU, accumulate on CPU
            gd, gh, gw = centers_zyx.shape[:3]
            tomogram_shape = (gd * sidelength, gh * sidelength, gw * sidelength)
            # Allocate output volume on CPU
            tomogram = torch.zeros(tomogram_shape, device='cpu', dtype=torch.float32)
            
            centers_flat, ps = einops.pack([centers_zyx_ang], "* zyx")
            total_patches = centers_flat.shape[0]
            
            # Pre-compute all grid indices
            patch_indices = torch.arange(total_patches)
            iz_all = patch_indices // (gh * gw)
            iy_all = (patch_indices % (gh * gw)) // gw
            ix_all = patch_indices % gw
            
            # Process and tile patches in batches
            batch_idx = 0
            for chunk in centers_flat.split(batch_size):
                # Reconstruct batch on GPU and transfer to CPU
                patches_batch = self.reconstruct_subvolume(chunk, sidelength).cpu()
                
                # Place each patch in the output volume using pre-computed indices
                for j in range(len(patches_batch)):
                    idx = batch_idx + j
                    iz, iy, ix = iz_all[idx], iy_all[idx], ix_all[idx]
                    
                    tomogram[
                        iz*sidelength:(iz+1)*sidelength,
                        iy*sidelength:(iy+1)*sidelength,
                        ix*sidelength:(ix+1)*sidelength
                    ] = patches_batch[j]
                
                batch_idx += len(patches_batch)
                
                # Free memory 
                del patches_batch
                if self.device != 'cpu':
                    torch.cuda.empty_cache()
        
        # Crop to final volume shape 
        tomogram = tomogram[:d, :h, :w]
        
        return tomogram
