import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "wan" / "utils" / "cam_utils.py"
SPEC = importlib.util.spec_from_file_location("cam_utils", MODULE_PATH)
CAM_UTILS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CAM_UTILS)


def test_interpolate_camera_intrinsics_tracks_dynamic_focal_length():
    src_indices = np.array([0.0, 4.0, 8.0])
    tgt_indices = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    intrinsics = torch.tensor(
        [
            [400.0, 420.0, 200.0, 120.0],
            [440.0, 460.0, 204.0, 124.0],
            [480.0, 500.0, 208.0, 128.0],
        ]
    )

    result = CAM_UTILS.interpolate_camera_intrinsics(
        src_indices, intrinsics, tgt_indices
    )

    expected = torch.tensor(
        [
            [400.0, 420.0, 200.0, 120.0],
            [420.0, 440.0, 202.0, 122.0],
            [440.0, 460.0, 204.0, 124.0],
            [460.0, 480.0, 206.0, 126.0],
            [480.0, 500.0, 208.0, 128.0],
        ]
    )
    torch.testing.assert_close(result, expected)
    assert result.dtype == intrinsics.dtype
    assert result.device == intrinsics.device


def test_interpolate_camera_intrinsics_preserves_constant_calibration():
    src_indices = np.arange(9, dtype=np.float64)
    tgt_indices = np.linspace(0, 8, 3)
    intrinsics = torch.tensor([[500.0, 501.0, 320.0, 240.0]]).repeat(9, 1)

    result = CAM_UTILS.interpolate_camera_intrinsics(
        src_indices, intrinsics, tgt_indices
    )

    torch.testing.assert_close(result, intrinsics[[0, 4, 8]])


def test_interpolate_camera_intrinsics_supports_bfloat16():
    src_indices = np.array([0.0, 2.0])
    tgt_indices = np.array([0.0, 1.0, 2.0])
    intrinsics = torch.tensor(
        [[400.0, 420.0, 200.0, 120.0], [440.0, 460.0, 204.0, 124.0]],
        dtype=torch.bfloat16,
    )

    result = CAM_UTILS.interpolate_camera_intrinsics(
        src_indices, intrinsics, tgt_indices
    )

    expected = torch.tensor(
        [
            [400.0, 420.0, 200.0, 120.0],
            [420.0, 440.0, 202.0, 122.0],
            [440.0, 460.0, 204.0, 124.0],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(result, expected)
    assert result.dtype == torch.bfloat16


def test_resample_camera_intrinsics_matches_standard_latent_grid():
    src_indices = np.arange(9, dtype=np.float64)
    tgt_indices = np.linspace(0, 8, 3)
    intrinsics = torch.arange(9, dtype=torch.float32)[:, None].repeat(1, 4)

    result = CAM_UTILS.resample_camera_intrinsics(
        src_indices, intrinsics, tgt_indices
    )

    torch.testing.assert_close(result, intrinsics[[0, 4, 8]])


def test_resample_camera_intrinsics_matches_fast_chunk_grid():
    src_indices = np.arange(25, dtype=np.float64)
    tgt_indices = np.linspace(0, 24, 6)
    intrinsics = torch.from_numpy(src_indices).float()[:, None].repeat(1, 4)

    result = CAM_UTILS.resample_camera_intrinsics(
        src_indices, intrinsics, tgt_indices
    )

    expected = torch.from_numpy(tgt_indices).float()[:, None].repeat(1, 4)
    torch.testing.assert_close(result, expected)


def test_resample_camera_intrinsics_repeats_singleton_calibration():
    intrinsics = torch.tensor([[500.0, 501.0, 320.0, 240.0]])

    result = CAM_UTILS.resample_camera_intrinsics(
        np.arange(9), intrinsics, np.linspace(0, 8, 3)
    )

    torch.testing.assert_close(result, intrinsics.repeat(3, 1))


def test_resample_camera_intrinsics_ignores_frames_beyond_pose_prefix():
    src_indices = np.arange(5, dtype=np.float64)
    intrinsics = torch.arange(7, dtype=torch.float32)[:, None].repeat(1, 4)

    result = CAM_UTILS.resample_camera_intrinsics(
        src_indices, intrinsics, np.array([0.0, 4.0])
    )

    torch.testing.assert_close(result, intrinsics[[0, 4]])


def test_resample_camera_intrinsics_rejects_short_calibration():
    with pytest.raises(ValueError, match="at least as many frames"):
        CAM_UTILS.resample_camera_intrinsics(
            np.arange(5), torch.zeros(4, 4), np.array([0.0, 4.0])
        )


def test_resample_camera_intrinsics_can_force_static_calibration():
    intrinsics = torch.tensor(
        [[500.0, 501.0, 320.0, 240.0], [600.0, 601.0, 321.0, 241.0]]
    )

    result = CAM_UTILS.resample_camera_intrinsics(
        np.arange(2),
        intrinsics,
        np.linspace(0, 1, 3),
        force_static=True,
    )

    torch.testing.assert_close(result, intrinsics[0:1].repeat(3, 1))


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 5)])
def test_interpolate_camera_intrinsics_rejects_invalid_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        CAM_UTILS.interpolate_camera_intrinsics(
            np.arange(shape[0]), torch.zeros(shape), np.array([0.0])
        )


def test_interpolate_camera_intrinsics_rejects_misaligned_frames():
    with pytest.raises(ValueError, match="same number of frames"):
        CAM_UTILS.interpolate_camera_intrinsics(
            np.arange(3), torch.zeros(2, 4), np.array([0.0])
        )
