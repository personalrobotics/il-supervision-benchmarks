"""PushT's T between its 8 keypoints and its rigid pose (x, y, theta), the coordinates CCIL generates in.

The keypoints are gym_pusht's: keypoint i = R(theta) LOCAL[i] + (x, y), with (x, y) the block body's position and
theta its angle. LOCAL is gym_pusht 0.1.8's PushTEnv block vertices in the body frame (read from the environment;
this construction reproduces its get_keypoints exactly). A pose is recovered by the closed-form 2D rigid fit;
theta is unwrapped along each episode so that a history's two slots are continuous.
"""
import numpy as np

LOCAL = np.array([[-60.0, 0.0], [60.0, 0.0], [60.0, 30.0], [-60.0, 30.0],
                  [-15.0, 30.0], [15.0, 30.0], [15.0, 120.0], [-15.0, 120.0]])


def keypoints(pose: np.ndarray) -> np.ndarray:
    """(N, 3) poses (x, y, theta) -> (N, 16) keypoints [x0, y0, x1, y1, ...]."""
    pose = np.asarray(pose, np.float64)
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    x = LOCAL[None, :, 0] * c[:, None] - LOCAL[None, :, 1] * s[:, None] + pose[:, 0:1]
    y = LOCAL[None, :, 0] * s[:, None] + LOCAL[None, :, 1] * c[:, None] + pose[:, 1:2]
    return np.stack([x, y], -1).reshape(len(pose), 16)


def fit(kp: np.ndarray) -> np.ndarray:
    """(N, 16) keypoints -> (N, 3) poses: the rigid motion of LOCAL closest to them in least squares."""
    p = np.asarray(kp, np.float64).reshape(len(kp), 8, 2)
    lc, pc = LOCAL.mean(0), p.mean(1)
    l, q = LOCAL - lc, p - pc[:, None]
    theta = np.arctan2((l[None, :, 0] * q[..., 1] - l[None, :, 1] * q[..., 0]).sum(1),
                       (l[None, :, 0] * q[..., 0] + l[None, :, 1] * q[..., 1]).sum(1))
    c, s = np.cos(theta), np.sin(theta)
    t = pc - np.stack([c * lc[0] - s * lc[1], s * lc[0] + c * lc[1]], -1)
    return np.concatenate([t, theta[:, None]], 1)


def episode_poses(kp: np.ndarray) -> np.ndarray:
    """An episode's frames of keypoints -> poses with theta unwrapped along the episode."""
    pose = fit(kp)
    pose[:, 2] = np.unwrap(pose[:, 2])
    return pose
