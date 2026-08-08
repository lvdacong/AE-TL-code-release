"""Visualise the frozen temperature-conditioned sensor-response proxy.

The figure is generated from the same model and diagnostic arrays used by the
formal temperature-conditioned sensor-response experiment.  It shows the
prescribed spatially correlated temperature conditions, the
temperature-dependent sensitivity, the additive thermal output, and the
resulting signed measurement change.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from AE_temperature_sensor_drift import (  # noqa: E402
    DIFFICULTY_TEMPERATURE_SPANS_C,
    LINEAR_GAIN_WEIGHT,
    LINEAR_OFFSET_WEIGHT,
    MAIN_TEMPERATURE_SPAN_C,
    QUADRATIC_GAIN_WEIGHT,
    QUADRATIC_OFFSET_WEIGHT,
    T_REF_C,
    build_temperature_sensor_drift_model,
)
from AE_train_model_auxiliary import FIG_DPI, PLOT_STYLE, apply_style  # noqa: E402


DATA_PATH = (
    SCRIPT_DIR
    / "AD_preprocess_datasets_output"
    / "health_original_2000"
    / "preprocessed_data_raw.npz"
)
OUTPUT_PATH = (
    SCRIPT_DIR
    / "AE_model_train_and_detect_output"
    / "Sensor_Drift"
    / "fig_sensor_drift.png"
)
LEVELS_OUTPUT_PATH = OUTPUT_PATH.with_name("fig_sd_sensor_drift_levels.png")
MECHANISM_OUTPUT_PATH = OUTPUT_PATH.with_name("fig_sd_temperature_mechanism.png")
N_CONDITIONS = 100
TEMPERATURE_LEVELS_C = tuple(
    int(value) for value in DIFFICULTY_TEMPERATURE_SPANS_C if value > 0
)


def _plot_envelope(ax: plt.Axes, x: np.ndarray, curves: np.ndarray, color: str) -> None:
    """Plot all fixed-channel curves plus their central envelope."""
    ax.plot(x, curves, color="#BDBDBD", linewidth=0.28, alpha=0.16)
    q05, q50, q95 = np.quantile(curves, [0.05, 0.50, 0.95], axis=1)
    ax.fill_between(x, q05, q95, color=color, alpha=0.22, linewidth=0)
    ax.plot(x, q50, color=color, linewidth=1.25, label="Median (5th--95th percentile)")


def main() -> None:
    with np.load(DATA_PATH) as data:
        healthy = np.asarray(data["V"][:N_CONDITIONS], dtype=np.float64)

    model = build_temperature_sensor_drift_model("original")
    shifted, diagnostics, _ = model.transform(
        healthy, temperature_span_c=MAIN_TEMPERATURE_SPAN_C
    )

    # Sort sensors only for the heat maps, so neighbouring columns follow the
    # longitudinal/vertical/transverse coordinate order rather than FE ID.
    coords = model.coordinates
    sensor_order = np.lexsort((coords[:, 1], coords[:, 2], coords[:, 0]))
    temperature = diagnostics["temperature_c"][:, sensor_order]
    signed_change = (
        (shifted.astype(np.float64) - healthy) / model.source_scale[None, :]
    )[:, sensor_order]

    temperature_grid = np.linspace(
        T_REF_C - MAIN_TEMPERATURE_SPAN_C,
        T_REF_C + MAIN_TEMPERATURE_SPAN_C,
        201,
    )
    theta = (temperature_grid - T_REF_C) / MAIN_TEMPERATURE_SPAN_C
    gain_curves = (
        1.0
        + LINEAR_GAIN_WEIGHT * theta[:, None] * model.a[None, :]
        + QUADRATIC_GAIN_WEIGHT * theta[:, None] ** 2 * model.b[None, :]
    )
    thermal_output_curves = (
        LINEAR_OFFSET_WEIGHT * theta[:, None] * model.c[None, :]
        + QUADRATIC_OFFSET_WEIGHT * theta[:, None] ** 2 * model.q[None, :]
    )

    style = dict(PLOT_STYLE)
    style.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
        }
    )
    apply_style(style)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), constrained_layout=True)

    ax = axes[0, 0]
    im_t = ax.imshow(
        temperature,
        aspect="auto",
        cmap="coolwarm",
        vmin=T_REF_C - MAIN_TEMPERATURE_SPAN_C,
        vmax=T_REF_C + MAIN_TEMPERATURE_SPAN_C,
        interpolation="nearest",
    )
    ax.set_xlabel("Sensor order by position")
    ax.set_ylabel("Temperature condition")
    ax.set_title("(a) Prescribed sensor temperature")
    cbar_t = fig.colorbar(im_t, ax=ax, pad=0.02, shrink=0.93)
    cbar_t.set_label("Temperature ($^{\\circ}$C)")

    ax = axes[0, 1]
    _plot_envelope(ax, temperature_grid, gain_curves, "#2B6DA1")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.7)
    ax.axvline(T_REF_C, color="#999999", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Temperature ($^{\\circ}$C)")
    ax.set_ylabel("Sensitivity factor")
    ax.set_title("(b) Temperature-dependent sensitivity")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 0]
    _plot_envelope(ax, temperature_grid, thermal_output_curves, "#C65D57")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.7)
    ax.axvline(T_REF_C, color="#999999", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Temperature ($^{\\circ}$C)")
    ax.set_ylabel("Thermal output / source scale")
    ax.set_title("(c) Additive thermal output")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    limit = float(np.quantile(np.abs(signed_change), 0.995))
    im_d = ax.imshow(
        signed_change,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    ax.set_xlabel("Sensor order by position")
    ax.set_ylabel("Temperature condition")
    ax.set_title("(d) Signed measurement change")
    cbar_d = fig.colorbar(im_d, ax=ax, pad=0.02, shrink=0.93)
    cbar_d.set_label("Change / source scale")

    for ax in axes.flat:
        ax.tick_params(direction="in", length=2.5)
        ax.grid(False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=FIG_DPI)
    plt.close(fig)
    print(OUTPUT_PATH)

    # Compact mechanism figure retained separately from the unified
    # five-level scenario-construction figure.
    fig_mech, axes_mech = plt.subplots(
        1, 2, figsize=(7.4, 2.8), constrained_layout=True
    )

    ax = axes_mech[0]
    _plot_envelope(ax, temperature_grid, gain_curves, "#2B6DA1")
    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.7)
    ax.axvline(T_REF_C, color="#999999", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Temperature ($^{\\circ}$C)")
    ax.set_ylabel("Sensitivity factor")
    ax.set_title("(a) Temperature-dependent sensitivity")

    ax = axes_mech[1]
    _plot_envelope(ax, temperature_grid, thermal_output_curves, "#C65D57")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.7)
    ax.axvline(T_REF_C, color="#999999", linestyle=":", linewidth=0.7)
    ax.set_xlabel("Temperature ($^{\\circ}$C)")
    ax.set_ylabel("Thermal output / source scale")
    ax.set_title("(b) Additive thermal output")

    handles, labels = axes_mech[0].get_legend_handles_labels()
    fig_mech.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=1,
        frameon=False,
        fontsize=7.5,
    )
    for ax in axes_mech:
        ax.tick_params(direction="in", length=2.5)
        ax.grid(False)

    fig_mech.savefig(MECHANISM_OUTPUT_PATH, dpi=FIG_DPI)
    plt.close(fig_mech)
    print(MECHANISM_OUTPUT_PATH)

    # Five non-zero temperature-amplitude levels use identical conditions,
    # sensor ordering, and a shared signed colour scale.  Only the prescribed
    # amplitude changes between panels.
    signed_change_levels = []
    for span_c in TEMPERATURE_LEVELS_C:
        shifted_level, _, _ = model.transform(
            healthy, temperature_span_c=float(span_c)
        )
        signed_change_levels.append(
            (
                (shifted_level.astype(np.float64) - healthy)
                / model.source_scale[None, :]
            )[:, sensor_order]
        )
    signed_change_levels = np.stack(signed_change_levels, axis=0)
    shared_limit = float(
        np.ceil(np.quantile(np.abs(signed_change_levels), 0.995) * 10.0) / 10.0
    )

    # Match the 1:5 panel geometry of the repair and relocation construction
    # figures.  The shared colour bar occupies the top legend strip, leaving
    # all five panel centres aligned at exact fifth-width intervals.
    fig_levels = plt.figure(figsize=(10.0, 2.4333333333))
    axes_levels = [
        fig_levels.add_axes([index / 5.0, 0.18, 0.20, 0.68])
        for index in range(len(TEMPERATURE_LEVELS_C))
    ]
    norm = TwoSlopeNorm(vmin=-shared_limit, vcenter=0.0, vmax=shared_limit)
    for ax, span_c, change in zip(
        axes_levels, TEMPERATURE_LEVELS_C, signed_change_levels
    ):
        heatmap = ax.imshow(
            change,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xlabel(
            rf"$\Theta={span_c}\,^{{\circ}}\mathrm{{C}}$", labelpad=8
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")

    colourbar_axis = fig_levels.add_axes([0.38, 0.925, 0.24, 0.025])
    colourbar = fig_levels.colorbar(
        heatmap, cax=colourbar_axis, orientation="horizontal"
    )
    colourbar.set_ticks([-shared_limit, 0.0, shared_limit])
    colourbar.ax.tick_params(direction="in", length=2.0, labelsize=6, pad=1)
    colourbar.ax.set_title("Signed change / source scale", fontsize=7, pad=1.5)

    fig_levels.savefig(LEVELS_OUTPUT_PATH, dpi=FIG_DPI)
    plt.close(fig_levels)
    print(LEVELS_OUTPUT_PATH)


if __name__ == "__main__":
    main()
