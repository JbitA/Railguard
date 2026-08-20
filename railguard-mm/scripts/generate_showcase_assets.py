from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def demo_series(minutes: float = 1.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * 10)
    t = np.arange(n, dtype=float) / 10.0
    speed = 5.5 + 1.1 * np.sin(t / 24.0) + 0.15 * rng.normal(size=n)
    base = 1.3 + 0.13 * speed
    event = np.zeros_like(t)
    for center in (18.0, 41.0):
        event += 2.3 * np.exp(-0.5 * ((t - center) / 0.8) ** 2)
    measured = np.maximum(0.05, base + 0.15 * np.sin(t * 1.7) + event + 0.08 * rng.normal(size=n))
    # A transparent integration-only reference: one-step persistence. It is not a trained model.
    persistence = np.concatenate([[np.nan], measured[:-1]])
    sensors = np.stack([
        np.maximum(0.01, measured * 0.94 + 0.02 * rng.normal(size=n)),
        np.maximum(0.01, measured + 0.02 * rng.normal(size=n)),
        np.maximum(0.01, measured * 1.07 + 0.02 * rng.normal(size=n)),
    ])
    sync = np.abs(rng.normal(2.5, 1.2, size=n))
    return t, measured, persistence, sensors, sync


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate clearly labeled synthetic integration-demo visuals for the README.")
    ap.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t, measured, persistence, sensors, sync = demo_series()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, measured, label="Measured vibration")
    ax.plot(t, persistence, label="1-step persistence reference")
    ax.set_xlabel("Synthetic demo time (s)")
    ax.set_ylabel("Vibration RMS (m/s²)")
    ax.set_title("Integration demo: measured signal and reference forecast")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "showcase_timeseries.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i in range(3):
        ax.plot(t, sensors[i], label=f"Vibration sensor {i}")
    ax.set_xlabel("Synthetic demo time (s)")
    ax.set_ylabel("RMS (m/s²)")
    ax.set_title("Integration demo: spatial vibration channels")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "showcase_spatial.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(t, sync)
    ax.axhline(20.0, linestyle="--", label="Example rejection threshold")
    ax.set_xlabel("Synthetic demo time (s)")
    ax.set_ylabel("Camera↔sensor sync error (ms)")
    ax.set_title("Integration demo: synchronization observability")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "showcase_sync.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"wrote showcase assets to {args.out_dir}")


if __name__ == "__main__":
    main()
