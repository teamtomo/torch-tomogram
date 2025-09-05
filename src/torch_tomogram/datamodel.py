from typing import Annotated, Optional
from datetime import datetime
from teamtomo_basemodel import BaseModelTeamTomo
from pydantic import Field, AliasPath, AfterValidator
import torch
import torch.nn.functional as F
from torch_affine_utils.transforms_3d import Rz, Ry, Rx, T


def tensor_check(x: float | torch.Tensor) -> torch.Tensor:
    if isinstance(x, float):
        return torch.tensor(x)  # convert float to 0-dim tensor
    elif isinstance(x, torch.Tensor) and x.numel() == 1:
        return x.squeeze()  # squeeze to a 0-dim tensor
    else:
        raise ValueError(f"{x} is neither a float nor a single valued tensor")


class Alignment(BaseModelTeamTomo):
    """Rigid body alignment parameters

    The alignment describes a forward projection model from the sample to
    the record exposure in the detector.
    """
    # rotation parameters
    z_rotation_degrees: Annotated[
        float | torch.Tensor, AfterValidator(tensor_check)
    ] = (
        Field(alias='tilt_axis_angle', default=torch.tensor(0.0))
    )
    y_rotation_degrees: Annotated[
        float | torch.Tensor, AfterValidator(tensor_check)
    ] = (
        Field(alias='tilt_angle', default=torch.tensor(0.0))
    )
    x_rotation_degrees: Annotated[
        float | torch.Tensor, AfterValidator(tensor_check)
    ] = (
        Field(alias='x_tilt', default=torch.tensor(0.0))
    )

    # global image shifts
    y_shift_px: Annotated[
        float | torch.Tensor, AfterValidator(tensor_check)
    ] = torch.tensor(0.0)
    x_shift_px: Annotated[
        float | torch.Tensor, AfterValidator(tensor_check)
    ] = torch.tensor(0.0)

    @property
    def projection_matrix(self) -> torch.Tensor:
        """Matrices that project points from 3D -> 2D."""
        shifts = torch.stack((self.y_shift_px, self.x_shift_px))
        shifts = F.pad(shifts, (1, 0), value=0)
        r0 = Rx(self.x_rotation_degrees, zyx=True)
        r1 = Ry(self.y_rotation_degrees, zyx=True)
        r2 = Rz(self.z_rotation_degrees, zyx=True)
        t3 = T(shifts)
        return t3 @ r2 @ r1 @ r0


class Optics(BaseModelTeamTomo):
    raise NotImplementedError


class Exposure(BaseModelTeamTomo):
    alignment: Alignment
    optics_group: Optional[Optics] = None
    collection_time: Optional[datetime] = None


class TiltSeries(BaseModelTeamTomo):
    image_stack: torch.Tensor
    tilt_image_data: list[Exposure] =(
        Field(description="A list of metadata around tilt images")
    )
