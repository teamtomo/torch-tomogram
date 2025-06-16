import torch

from torch_tomogram import Tomogram


def test_projection_matrices():
    """Test that projection matrices are computed correctly."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations
    images = torch.zeros((3, 10, 10))  # 3 tilts, 10x10 images

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
    )

    # Get projection matrices
    matrices = tomogram.projection_matrices

    # Check shape and type
    assert matrices.shape == (3, 4, 4)  # 3 tilts, 4x4 matrices
    assert matrices.dtype == torch.float32


def test_project_points():
    """Test that points can be projected from 3D to 2D."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations
    images = torch.zeros((3, 10, 10))  # 3 tilts, 10x10 images

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
    )

    # Create a single 3D point at the origin
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]])

    # Project the point
    projected_yx = tomogram.project_points(points_zyx)

    # Check shape and type
    assert projected_yx.shape == (1, 3, 2)  # 1 point, 3 tilts, 2D coordinates
    assert projected_yx.dtype == torch.float32

    # For a point at the origin with no translations, the y-coordinate should be 0
    # for all tilts, and the x-coordinate should depend on the tilt angle
    assert torch.allclose(projected_yx[0, :, 0], torch.tensor([0.0, 0.0, 0.0]))


def test_extract_particle_tilt_series():
    """Test that particle tilt series can be extracted."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations

    # Create simple images with a pattern
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0  # Create a small square in the center

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
    )

    # Create a single 3D point at the origin
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]])

    # Extract particle tilt series
    particle_tilt_series = tomogram.extract_particle_tilt_series(
        points_zyx, sidelength=8
    )

    # Check shape and type
    assert particle_tilt_series.shape == (1, 3, 8, 8)  # 1 point, 3 tilts, 8x8 images
    assert particle_tilt_series.dtype == torch.float32


def test_reconstruct_subvolume():
    """Test that a subvolume can be reconstructed."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations

    # Create simple images with a pattern
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0  # Create a small square in the center

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
    )

    # Create a single 3D point at the origin
    point_zyx = torch.tensor([0.0, 0.0, 0.0])

    # Reconstruct subvolume
    subvolume = tomogram.reconstruct_subvolume(point_zyx, sidelength=8)

    # Check shape and type
    assert subvolume.shape == (8, 8, 8)  # 8x8x8 volume
    assert subvolume.dtype == torch.float32


def test_reconstruct_tomogram():
    """Test that a tomogram can be reconstructed."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations

    # Create simple images with a pattern
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0  # Create a small square in the center

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
    )

    # Reconstruct a small tomogram
    volume = tomogram.reconstruct_tomogram((16, 16, 16), sidelength=8)

    # Check shape and type
    assert volume.shape == (16, 16, 16)  # 16x16x16 volume
    assert volume.dtype == torch.float32
