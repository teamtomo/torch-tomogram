"""Tomogram reconstruction in pytorch."""

import einops
import torch
import torch.nn.functional as F
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Ry, Rz, T
from torch_fourier_slice import backproject_2d_to_3d
from torch_grid_utils import dft_center
from torch_subpixel_crop import subpixel_crop_2d


class Tomogram:
    """Tomogram class that enables reconstruction position querying."""

    def __init__(
        self,
        tilt_angles: torch.Tensor,
        tilt_axis_angle: torch.Tensor,
        sample_translations: torch.Tensor,
        images: torch.Tensor | None = None,  # (b, h, w)
    ):
        self.images = torch.tensor(images).float()
        self.tilt_angles = torch.tensor(tilt_angles).float()
        self.tilt_axis_angle = torch.tensor(tilt_axis_angle).float()
        self.sample_translations = torch.tensor(sample_translations).float()

    @property
    def projection_matrices(self) -> torch.Tensor:
        """Matrices that project points from 3D -> 2D."""
        shifts_3d = F.pad(self.sample_translations, (1, 0), value=0)
        r0 = Ry(self.tilt_angles, zyx=True)
        r1 = Rz(self.tilt_axis_angle, zyx=True)
        t2 = T(shifts_3d)
        return t2 @ r1 @ r0

    def project_points(self, points_zyx: torch.Tensor) -> torch.Tensor:
        """Project points from 3D to 2D.

        - points are 3D zyx coordinates
        - points are positions relative to center of tomogram
        - projected 2D points are relative to center of 2D image
        """
        points_zyx = torch.tensor(points_zyx).float()
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
        self, points_zyx: torch.Tensor, sidelength: int
    ) -> torch.Tensor:
        """Extract a subtilt-series at a 3D location in the sample."""
        projected_yx = self.project_points(points_zyx)
        projected_yx += dft_center(self.images.shape[-2:], rfft=False, fftshift=True)
        images = subpixel_crop_2d(
            image=self.images,
            positions=projected_yx,
            sidelength=sidelength,
        )
        return images

    def reconstruct_subvolume(
        self, point_zyx: torch.Tensor, sidelength: int
    ) -> torch.Tensor:
        """Reconstruct a subvolume at a 3D location in the sample."""
        point_zyx = torch.tensor(point_zyx).float()
        point_zyx = point_zyx.reshape((-1, 3))
        rotation_matrices = self.projection_matrices[:, :3, :3]
        rotation_matrices = torch.linalg.pinv(rotation_matrices)
        particle_tilt_series = self.extract_particle_tilt_series(
            point_zyx, sidelength=sidelength
        )
        volume = backproject_2d_to_3d(
            images=particle_tilt_series[0],
            rotation_matrices=rotation_matrices,
            zyx_matrices=True,
        )
        return volume

    def reconstruct_tomogram(
        self, volume_shape: tuple[int, int, int], sidelength: int
    ) -> torch.Tensor:
        """Reconstruct the full tomogram by tiling the positions in 3D."""
        d, h, w = volume_shape
        r = sidelength // 2

        # setup grid points
        z = torch.arange(start=r, end=d + r, step=sidelength) - d // 2
        y = torch.arange(start=r, end=h + r, step=sidelength) - h // 2
        x = torch.arange(start=r, end=w + r, step=sidelength) - w // 2

        # allocate whole volume
        tomogram = torch.zeros(size=volume_shape, dtype=torch.float32)

        for _z in z:
            for _y in y:
                for _x in x:
                    zyx = torch.tensor([_z, _y, _x]).float()
                    subvolume = self.reconstruct_subvolume(zyx, sidelength=sidelength)
                    _d, _h, _w = zyx + torch.tensor(volume_shape) // 2
                    print(_d, _h, _w)
                    _d, _h, _w = int(_d), int(_h), int(_w)
                    d_min, d_max = _d - r, _d + r
                    h_min, h_max = _h - r, _h + r
                    w_min, w_max = _w - r, _w + r
                    tomogram[d_min:d_max, h_min:h_max, w_min:w_max] = subvolume
        return tomogram
