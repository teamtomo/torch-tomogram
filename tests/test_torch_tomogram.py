import pytest
import torch

from torch_tomogram import Tomogram

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="can only move device with CUDA support"
)
def test_device_move():
    """Test that tensors can move to other device."""
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
        pixel_spacing=1.0,
    )
    assert "cpu" == str(tomogram.images.device)

    tomogram.to("cuda")

    # Check shape and type
    assert "cuda" in str(tomogram.images.device)
    assert "cuda" in str(tomogram.device)
    assert "cuda" in str(tomogram.tilt_angles.device)
    assert "cuda" in str(tomogram.tilt_axis_angle.device)
    assert "cuda" in str(tomogram.sample_translations.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_projection_matrices(device):
    """Test that projection matrices are computed correctly."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations
    images = torch.zeros((3, 10, 10), device=device)  # 3 tilts, 10x10 images

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        pixel_spacing=1.0,
        device=device,
    )

    # Get projection matrices
    matrices = tomogram.projection_matrices

    # Check shape and type
    assert matrices.shape == (3, 4, 4)  # 3 tilts, 4x4 matrices
    assert matrices.dtype == torch.float32
    assert device in str(matrices.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_project_points(device):
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
        pixel_spacing=1.0,
        device=device,
    )

    # Create a single 3D point at the origin
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # Project the point
    projected_yx = tomogram.project_points(points_zyx)

    # Check shape and type
    assert projected_yx.shape == (1, 3, 2)  # 1 point, 3 tilts, 2D coordinates
    assert projected_yx.dtype == torch.float32
    assert device in str(projected_yx.device)

    # For a point at the origin with no translations, the y-coordinate should be 0
    # for all tilts, and the x-coordinate should depend on the tilt angle
    assert torch.allclose(
        projected_yx[0, :, 0], torch.tensor([0.0, 0.0, 0.0], device=device)
    )


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_extract_particle_tilt_series(device):
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
        pixel_spacing=1.0,
        device=device,
    )

    # Create a single 3D point at the origin
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # Extract particle tilt series in real space
    particle_tilt_series = tomogram.extract_particle_tilt_series(
        points_zyx, sidelength=8, return_rfft=False
    )

    # Check shape and type
    assert particle_tilt_series.shape == (1, 3, 8, 8)  # 1 point, 3 tilts, 8x8 images
    assert particle_tilt_series.dtype == torch.float32
    assert device in str(particle_tilt_series.device)
    
    # Also test Fourier space extraction
    particle_tilt_series_rfft = tomogram.extract_particle_tilt_series(
        points_zyx, sidelength=8, return_rfft=True
    )
    assert particle_tilt_series_rfft.shape == (1, 3, 8, 5)  # rfft: width = 8//2 + 1 = 5
    assert particle_tilt_series_rfft.dtype == torch.complex64


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_reconstruct_subvolume(device):
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
        pixel_spacing=1.0,
        device=device,
    )

    # Create a single 3D point at the origin
    point_zyx = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Reconstruct subvolume
    subvolume = tomogram.reconstruct_subvolume(point_zyx, sidelength=8)

    # Check shape and type
    assert subvolume.shape == (8, 8, 8)  # 8x8x8 volume
    assert subvolume.dtype == torch.float32
    assert device in str(subvolume.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_reconstruct_tomogram(device):
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
        pixel_spacing=1.0,
        device=device,
    )

    # Reconstruct a small tomogram
    volume = tomogram.reconstruct_tomogram((16, 16, 16), sidelength=8)

    # Check shape and type
    assert volume.shape == (16, 16, 16)  # 16x16x16 volume
    assert volume.dtype == torch.float32
    assert device in str(volume.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_reconstruct_subvolume_rank_polymorphic(device):
    """Test that reconstruct_subvolume handles different input shapes (rank-polymorphic)."""
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0
    
    # initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        pixel_spacing=1.0,
        device=device,
    )
    
    # Test single point (3,) -> (d, h, w)
    point = torch.tensor([0.0, 0.0, 0.0])
    subvolume = tomogram.reconstruct_subvolume(point, sidelength=8)
    assert subvolume.shape == (8, 8, 8)
    
    # Test batch (N, 3) -> (N, d, h, w)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    subvolumes = tomogram.reconstruct_subvolume(points, sidelength=8)
    assert subvolumes.shape == (2, 8, 8, 8)
    
    # Test 2D grid (h, w, 3) -> (h, w, d, d, d)
    grid_2d = torch.zeros(2, 3, 3)
    subvolumes_grid = tomogram.reconstruct_subvolume(grid_2d, sidelength=8)
    assert subvolumes_grid.shape == (2, 3, 8, 8, 8)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_reconstruct_tomogram_batch_size(device):
    """Test that reconstruct_tomogram works with batch_size parameter."""
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0
    
    # initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        pixel_spacing=1.0,
        device=device,
    )
    
    # Reconstruct without batch_size
    recon_no_batch = tomogram.reconstruct_tomogram((16, 16, 16), sidelength=8)
    
    # Reconstruct with batch_size
    recon_with_batch = tomogram.reconstruct_tomogram((16, 16, 16), sidelength=8, batch_size=2)
    
    # Both should produce same shape
    assert recon_no_batch.shape == (16, 16, 16)
    assert recon_with_batch.shape == (16, 16, 16)
    
    # Results should be very similar 
    diff = torch.abs(recon_no_batch - recon_with_batch.to(recon_no_batch.device)).max()
    assert diff == 0.0
