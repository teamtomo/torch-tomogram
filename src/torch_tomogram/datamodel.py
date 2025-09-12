from typing import Annotated, Optional
from datetime import datetime
from teamtomo_basemodel import BaseModelTeamTomo, ExcludedTensor
from pydantic import Field, AliasPath


class RigidBodyAlignment(BaseModelTeamTomo):
    """Rigid body alignment parameters

    The alignment describes a forward projection model from the sample to
    the record exposure in the detector.
    """
    # rotation parameters
    z_rotation_degrees: float = Field(alias='tilt_axis_angle', default=0.0)
    y_rotation_degrees: float = Field(alias='tilt_angle', default=0.0)
    x_rotation_degrees: float = Field(alias='x_tilt', default=0.0)

    # global image shifts
    y_shift_angstrom: float = Field(default=0.0)
    x_shift_angstrom: float = Field(default=0.0)


class Optics(BaseModelTeamTomo):
    defocus_u: float
    defocus_v: float
    # etc...


class Exposure(BaseModelTeamTomo):
    collection_time: Optional[datetime] = None
    accumulated_dose: float = Field(default=0.0, gt=0.0)  # in e- per A^2

    alignment: RigidBodyAlignment
    optics: Optics


class ExposureSeries(BaseModelTeamTomo):
    pixel_spacing_angstrom: float = Field(gt=0.0)
    image_stack: ExcludedTensor
    exposure_list: list[Exposure] =(
        Field(description="A list of metadata around tilt images")
    )
