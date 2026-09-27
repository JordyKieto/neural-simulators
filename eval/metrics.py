"""Frame metrics shared by the CARLA and TinySkies evaluations. Frames are float arrays in [0,1]."""

import numpy as np
import tensorflow as tf


def mae(a, b):
    return np.abs(a - b).reshape(len(a), -1).mean(axis=1)


def psnr(a, b):
    mse = np.square(a - b).reshape(len(a), -1).mean(axis=1)
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-10))


def ssim(a, b):
    return tf.image.ssim(tf.constant(a, tf.float32), tf.constant(b, tf.float32), max_val=1.0).numpy()


def sharpness(frames):
    """Variance of the Laplacian of luminance: high-frequency detail (drops as frames blur)."""
    y = frames @ np.array([0.299, 0.587, 0.114], np.float32)
    lap = -4 * y[:, 1:-1, 1:-1] + y[:, :-2, 1:-1] + y[:, 2:, 1:-1] + y[:, 1:-1, :-2] + y[:, 1:-1, 2:]
    return lap.reshape(len(frames), -1).var(axis=1)


def frame_change(prev, curr):
    """Mean absolute change between consecutive frames: how fast the image content is moving."""
    return mae(prev, curr)


def horizontal_shift(prev, curr, band=(0.35, 0.75)):
    """Estimated horizontal image translation (pixels, + = content moves right) between two
    frames, by phase correlation on a horizontal band of rows. Used to test whether turn inputs
    rotate the view in the expected direction."""
    h = prev.shape[1]
    r0, r1 = int(band[0] * h), int(band[1] * h)
    lum = np.array([0.299, 0.587, 0.114], np.float32)
    shifts = []
    for p, c in zip(prev, curr):
        a = (p[r0:r1] @ lum).mean(axis=0)
        b = (c[r0:r1] @ lum).mean(axis=0)
        a, b = a - a.mean(), b - b.mean()
        win = np.hanning(len(a))
        fa, fb = np.fft.fft(a * win), np.fft.fft(b * win)
        r = fb * np.conj(fa)
        corr = np.fft.ifft(r / np.maximum(np.abs(r), 1e-9)).real
        k = int(np.argmax(corr))
        shifts.append(k if k <= len(a) // 2 else k - len(a))
    return np.asarray(shifts, np.float32)
