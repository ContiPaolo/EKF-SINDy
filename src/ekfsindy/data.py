"""Data utilities: noise, time-delay embeddings and the shear-building dataset loader."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd


def add_noise_with_snr(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise with signal-to-noise ratio ``snr_db`` (in decibel).

    The SNR is the ratio between the mean square of the whole ``signal`` and the noise
    variance. Noise is drawn from NumPy's global generator (seed it with ``np.random.seed``).
    """
    power_db = 10 * np.log10(np.mean(np.square(signal)))
    noise_var = 10 ** ((power_db - snr_db) / 10)
    return signal + np.random.normal(0, np.sqrt(noise_var), signal.shape)


def hankel(vector: np.ndarray, length: int, shift: int = 1) -> np.ndarray:
    """Hankel matrix ``H[i, j] = vector[i * shift + j]`` with ``length`` rows."""
    vector = np.asarray(vector).ravel()
    n_cols = len(vector) - length * shift + 1
    return np.lib.stride_tricks.sliding_window_view(vector, n_cols)[::shift][:length].copy()


class TimeDelayEmbedding:
    """Time-delay coordinates of a scalar signal from the SVD of its Hankel matrix.

    ``fit`` stacks the Hankel matrices of the training signals and keeps the leading
    ``n_coordinates`` modes; ``transform`` maps a signal to ``H^T U S^-1`` and
    ``observation_matrix`` maps coordinates back to the first delayed sample.
    """

    def __init__(self, length: int, n_coordinates: int, shift: int = 1, n_components: int = 16):
        self.length, self.n_coordinates = length, n_coordinates
        self.shift, self.n_components = shift, n_components

    def fit(self, signals: Sequence[np.ndarray]) -> TimeDelayEmbedding:
        H = np.hstack([hankel(s, self.length, self.shift) for s in signals])
        # random_state=None draws from NumPy's global generator, as the original notebooks did
        U, S, _ = randomized_svd(H, n_components=self.n_components)
        self.singular_values = S
        self.U = U[:, :self.n_coordinates]
        self.S = S[:self.n_coordinates]
        return self

    def n_samples(self, n_time: int) -> int:
        """Number of delay-coordinate samples obtained from a signal of length ``n_time``."""
        return n_time - self.length * self.shift + 1

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Delay coordinates of ``signal``, shape ``(n_samples, n_coordinates)``."""
        return hankel(signal, self.length, self.shift).T @ (self.U / self.S)

    def inverse_transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Reconstructed Hankel rows ``coordinates @ (U S)^T``, shape ``(n_samples, length)``."""
        return coordinates @ (self.U * self.S).T

    @property
    def observation_matrix(self) -> np.ndarray:
        """``C`` such that ``C @ coordinates`` is the (first) delayed sample, shape ``(1, n)``."""
        return (self.U * self.S)[:1, :]


def data_dir() -> Path:
    """Dataset folder: ``$EKFSINDY_DATA`` if set, otherwise ``data/`` in the repository."""
    env = os.environ.get("EKFSINDY_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"


def read_dof_csv(
    prefix: str,
    split: str,
    seq_len: int,
    n_channels: int,
    root: str | os.PathLike | None = None,
) -> np.ndarray:
    """Load ``<prefix>dof_<i>_<split>.csv`` for ``i = 1..n_channels`` from the shear dataset.

    Each file is a single column of concatenated instances of length ``seq_len``; the result
    has shape ``(n_instances, seq_len, n_channels)``.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    root = Path(root) if root is not None else data_dir()
    channels = []
    for i in range(1, n_channels + 1):
        column = np.loadtxt(root / f"{prefix}dof_{i}_{split}.csv")
        n_instances = len(column) // seq_len
        channels.append(column[: n_instances * seq_len].reshape(n_instances, seq_len))
    return np.stack(channels, axis=-1)
