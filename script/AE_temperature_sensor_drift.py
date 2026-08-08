"""Deterministic temperature-conditioned sensor-response drift.

This module is the single source of truth for the temperature-sensor shift
used by the AE-TL experiments.  It deliberately separates

* parameter construction (source healthy data and sensor coordinates only),
* transformation of an arbitrary healthy or damaged data matrix, and
* optional persistence of the transformed data and complete diagnostics.

No target or damaged data are used to estimate any model parameter.  Calling
``transform`` on healthy and damaged arrays with the same measurement layout
reuses the same 100-condition sequence in the same cyclic order.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
AC_OUTPUT = SCRIPT_DIR / "AC_convert_and_extract_output"
AD_OUTPUT = SCRIPT_DIR / "AD_preprocess_datasets_output"

DEFAULT_VTU_PATH = AC_OUTPUT / "whole_from_inp.vtu"
DEFAULT_ID_MAPPING_PATH = AC_OUTPUT / "abaqus_id_to_vtu_index.csv"
DEFAULT_SOURCE_HEALTH_PATH = (
    AD_OUTPUT / "health_original_2000" / "preprocessed_data_raw.npz"
)

T_REF_C = 20.0
MAIN_TEMPERATURE_SPAN_C = 20.0
DIFFICULTY_TEMPERATURE_SPANS_C = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)
N_TEMPERATURE_CONDITIONS = 100
CONDITION_SEED = 20260806
COEFFICIENT_SEED = 42
SOURCE_TRAIN_SAMPLES = 1800

GLOBAL_TEMPERATURE_WEIGHT = 0.7
SPATIAL_TEMPERATURE_WEIGHT = 0.3
LINEAR_GAIN_WEIGHT = 0.15
QUADRATIC_GAIN_WEIGHT = 0.045
LINEAR_OFFSET_WEIGHT = 0.15
QUADRATIC_OFFSET_WEIGHT = 0.045


def _as_float_matrix(values: np.ndarray, name: str, columns: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array, got {array.shape}")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(
            f"{name} has {array.shape[1]} columns; expected {columns}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _latin_hypercube(n_samples: int, n_dimensions: int, seed: int) -> np.ndarray:
    """Return a reproducible Latin-hypercube design in [-1, 1]."""
    if n_samples <= 0 or n_dimensions <= 0:
        raise ValueError("Latin-hypercube dimensions must be positive")
    rng = np.random.RandomState(seed)
    design = np.empty((n_samples, n_dimensions), dtype=np.float64)
    for column in range(n_dimensions):
        bins = (np.arange(n_samples, dtype=np.float64) + rng.random_sample(n_samples))
        design[:, column] = bins[rng.permutation(n_samples)] / n_samples
    return 2.0 * design - 1.0


def make_temperature_conditions(
    n_conditions: int = N_TEMPERATURE_CONDITIONS,
    seed: int = CONDITION_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct fixed Latin-hypercube global levels ``u`` and gradients ``g``.

    The first LHS dimension is the global temperature level ``u``.  The next
    three dimensions form the spatial-gradient vector ``g``.  All components
    lie in [-1, 1].
    """
    design = _latin_hypercube(n_conditions, 4, seed)
    return design[:, 0].copy(), design[:, 1:4].copy()


def make_channel_coefficients(
    n_channels: int,
    seed: int = COEFFICIENT_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate fixed channel coefficients ``a``, ``b``, ``c`` and ``q``."""
    if n_channels <= 0:
        raise ValueError("n_channels must be positive")
    rng = np.random.RandomState(seed)
    a = np.clip(1.0 + 0.3 * rng.standard_normal(n_channels), 0.4, 1.6)
    b = np.clip(rng.standard_normal(n_channels), -2.0, 2.0) / 2.0
    c = np.clip(rng.standard_normal(n_channels), -2.0, 2.0) / 2.0
    q = np.clip(rng.standard_normal(n_channels), -2.0, 2.0) / 2.0
    return a, b, c, q


def resolve_measurement_id_path(
    measurement_layout: str | Path = "original",
    ac_output: Path = AC_OUTPUT,
) -> Path:
    """Resolve ``original``, ``offset_count_N`` or an explicit measurement CSV."""
    candidate = Path(measurement_layout)
    if candidate.is_file():
        return candidate.resolve()

    name = str(measurement_layout)
    if name.lower() == "original":
        filename = "measures_ID_original.csv"
    elif name.lower().endswith(".csv"):
        filename = name
    elif name.startswith("measures_ID_"):
        filename = f"{name}.csv"
    else:
        filename = f"measures_ID_{name}.csv"

    path = ac_output / filename
    if not path.is_file():
        raise FileNotFoundError(f"Measurement-ID file not found: {path}")
    return path.resolve()


def load_measurement_ids(measurement_id_path: str | Path) -> np.ndarray:
    """Load measurement element IDs in channel order from a one-column CSV."""
    path = Path(measurement_id_path)
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8-sig")
    if data.size == 0:
        raise ValueError(f"No measurement IDs found in {path}")
    if data.dtype.names is None or len(data.dtype.names) != 1:
        raise ValueError(f"Expected a one-column measurement-ID CSV: {path}")
    ids = np.atleast_1d(data[data.dtype.names[0]]).astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Measurement IDs are not unique in {path}")
    return ids


def load_measurement_coordinates(
    measurement_id_path: str | Path,
    vtu_path: str | Path = DEFAULT_VTU_PATH,
    id_mapping_path: str | Path = DEFAULT_ID_MAPPING_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Return measurement IDs and their VTU cell-centre coordinates."""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on runtime packaging
        raise RuntimeError(
            "pyvista is required only when loading coordinates from the VTU file"
        ) from exc

    measurement_ids = load_measurement_ids(measurement_id_path)
    mapping = np.genfromtxt(
        id_mapping_path, delimiter=",", names=True, dtype=None, encoding="utf-8-sig"
    )
    if mapping.dtype.names is None or set(mapping.dtype.names) != {
        "abaqus_id", "vtu_index"
    }:
        raise ValueError(f"Unexpected ID-mapping schema: {id_mapping_path}")
    id_to_index = {
        int(abaqus_id): int(vtu_index)
        for abaqus_id, vtu_index in zip(mapping["abaqus_id"], mapping["vtu_index"])
    }

    missing = [int(element_id) for element_id in measurement_ids if int(element_id) not in id_to_index]
    if missing:
        raise KeyError(f"Measurement IDs missing from VTU mapping: {missing[:10]}")

    mesh = pv.read(str(vtu_path))
    cell_centres = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    indices = np.asarray([id_to_index[int(element_id)] for element_id in measurement_ids])
    if np.any(indices < 0) or np.any(indices >= len(cell_centres)):
        raise IndexError("The ID mapping contains VTU cell indices outside the mesh")
    return measurement_ids, cell_centres[indices]


def normalise_measurement_coordinates(
    coordinates: np.ndarray,
    reference_coordinates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise coordinates to a shared, component-wise [-1, 1] frame."""
    coordinates = _as_float_matrix(coordinates, "coordinates", columns=3)
    reference = coordinates if reference_coordinates is None else _as_float_matrix(
        reference_coordinates, "reference_coordinates", columns=3
    )
    centre = 0.5 * (reference.min(axis=0) + reference.max(axis=0))
    half_range = 0.5 * (reference.max(axis=0) - reference.min(axis=0))
    safe_half_range = np.where(half_range > 0.0, half_range, 1.0)
    rbar = np.clip((coordinates - centre) / safe_half_range, -1.0, 1.0)
    return rbar, centre, half_range


def load_source_scale(
    source_health_path: str | Path = DEFAULT_SOURCE_HEALTH_PATH,
    source_train_samples: int = SOURCE_TRAIN_SAMPLES,
) -> np.ndarray:
    """Compute channel scales solely from the first 1800 source healthy rows."""
    if source_train_samples <= 0:
        raise ValueError("source_train_samples must be positive")
    with np.load(source_health_path) as data:
        if "V" not in data:
            raise KeyError(f"Dataset has no 'V' array: {source_health_path}")
        source = _as_float_matrix(data["V"], "source healthy data")
    if source.shape[0] < source_train_samples:
        raise ValueError(
            f"Source healthy data has {source.shape[0]} rows; "
            f"{source_train_samples} are required"
        )
    return source[:source_train_samples].std(axis=0, ddof=0)


@dataclass(frozen=True)
class TemperatureSensorDriftModel:
    """All fixed parameters needed to transform healthy or damaged data."""

    measurement_ids: np.ndarray
    coordinates: np.ndarray
    normalised_coordinates: np.ndarray
    coordinate_reference_centre: np.ndarray
    coordinate_reference_half_range: np.ndarray
    source_scale: np.ndarray
    condition_u: np.ndarray
    condition_g: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    q: np.ndarray
    measurement_layout: str
    measurement_id_path: str
    coordinate_reference_layout: str
    coordinate_reference_id_path: str
    source_health_path: str

    @property
    def n_channels(self) -> int:
        return int(self.measurement_ids.size)

    def transform(
        self,
        values: np.ndarray,
        temperature_span_c: float = MAIN_TEMPERATURE_SPAN_C,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """Apply the frozen response model and return data, diagnostics, config."""
        x = _as_float_matrix(values, "values", columns=self.n_channels)
        span = float(temperature_span_c)
        if not np.isfinite(span) or span < 0.0:
            raise ValueError("temperature_span_c must be finite and non-negative")

        condition_index = np.arange(x.shape[0], dtype=np.int64) % len(self.condition_u)
        u = self.condition_u[condition_index]
        g = self.condition_g[condition_index]
        g_l1 = np.abs(g).sum(axis=1)
        if np.any(g_l1 <= 0.0):
            raise ValueError("A temperature-gradient condition has zero L1 norm")

        spatial = np.einsum("ni,di->nd", g, self.normalised_coordinates) / g_l1[:, None]
        tau = GLOBAL_TEMPERATURE_WEIGHT * u[:, None] + SPATIAL_TEMPERATURE_WEIGHT * spatial
        temperature_c = T_REF_C + span * tau
        theta = (temperature_c - T_REF_C) / MAIN_TEMPERATURE_SPAN_C

        gain = (
            1.0
            + LINEAR_GAIN_WEIGHT * self.a[None, :] * theta
            + QUADRATIC_GAIN_WEIGHT * self.b[None, :] * theta**2
        )
        additive = self.source_scale[None, :] * (
            LINEAR_OFFSET_WEIGHT * self.c[None, :] * theta
            + QUADRATIC_OFFSET_WEIGHT * self.q[None, :] * theta**2
        )
        transformed = gain * x + additive

        diagnostics: dict[str, Any] = {
            "measurement_ids": self.measurement_ids.copy(),
            "coordinates": self.coordinates.copy(),
            "normalised_coordinates": self.normalised_coordinates.copy(),
            "source_scale": self.source_scale.copy(),
            "condition_index": condition_index,
            "condition_u": self.condition_u.copy(),
            "condition_g": self.condition_g.copy(),
            "sample_u": u,
            "sample_g": g,
            "spatial_term": spatial,
            "tau": tau,
            "theta": theta,
            "temperature_c": temperature_c,
            "gain": gain,
            "additive": additive,
            "a": self.a.copy(),
            "b": self.b.copy(),
            "c": self.c.copy(),
            "q": self.q.copy(),
        }
        config = self.config(temperature_span_c=span, n_samples=x.shape[0])
        return transformed.astype(np.float32), diagnostics, config

    def config(self, temperature_span_c: float, n_samples: int) -> dict[str, Any]:
        return {
            "model": "temperature-conditioned sensor-response drift",
            "equation": (
                "tau=0.7*u+0.3*(g^T*rbar/||g||_1); "
                "T=T_ref+temperature_span*tau; "
                "theta=(T-T_ref)/20; "
                "y=(1+0.15*a*theta+0.045*b*theta^2)*x+"
                "source_scale*(0.15*c*theta+0.045*q*theta^2)"
            ),
            "t_ref_c": T_REF_C,
            "temperature_span_c": float(temperature_span_c),
            "main_temperature_span_c": MAIN_TEMPERATURE_SPAN_C,
            "difficulty_temperature_spans_c": list(DIFFICULTY_TEMPERATURE_SPANS_C),
            "n_temperature_conditions": int(len(self.condition_u)),
            "condition_seed": CONDITION_SEED,
            "coefficient_seed": COEFFICIENT_SEED,
            "coefficient_rng": "numpy.random.RandomState(MT19937)",
            "condition_design": "four-dimensional Latin hypercube mapped to [-1, 1]",
            "condition_reuse": "sample_index modulo 100",
            "n_samples": int(n_samples),
            "n_channels": self.n_channels,
            "measurement_layout": self.measurement_layout,
            "measurement_id_path": self.measurement_id_path,
            "coordinate_reference_layout": self.coordinate_reference_layout,
            "coordinate_reference_id_path": self.coordinate_reference_id_path,
            "coordinate_reference_centre": self.coordinate_reference_centre.tolist(),
            "coordinate_reference_half_range": self.coordinate_reference_half_range.tolist(),
            "source_health_path": self.source_health_path,
            "source_scale_rows": [0, SOURCE_TRAIN_SAMPLES],
            "source_scale_statistic": "population standard deviation (ddof=0)",
            "parameter_data_policy": (
                "coefficients are seed-fixed; scale uses only the first 1800 "
                "health_original rows; no target or damaged samples determine parameters"
            ),
        }


def build_temperature_sensor_drift_model(
    measurement_layout: str | Path = "original",
    *,
    coordinate_reference_layout: str | Path = "original",
    source_health_path: str | Path = DEFAULT_SOURCE_HEALTH_PATH,
    vtu_path: str | Path = DEFAULT_VTU_PATH,
    id_mapping_path: str | Path = DEFAULT_ID_MAPPING_PATH,
) -> TemperatureSensorDriftModel:
    """Build a frozen model for original or offset measurement IDs."""
    measurement_path = resolve_measurement_id_path(measurement_layout)
    reference_path = resolve_measurement_id_path(coordinate_reference_layout)
    measurement_ids, coordinates = load_measurement_coordinates(
        measurement_path, vtu_path, id_mapping_path
    )
    _, reference_coordinates = load_measurement_coordinates(
        reference_path, vtu_path, id_mapping_path
    )
    rbar, centre, half_range = normalise_measurement_coordinates(
        coordinates, reference_coordinates
    )
    source_scale = load_source_scale(source_health_path)
    if source_scale.size != measurement_ids.size:
        raise ValueError(
            f"Source scale has {source_scale.size} channels; measurement layout "
            f"has {measurement_ids.size}"
        )
    condition_u, condition_g = make_temperature_conditions()
    a, b, c, q = make_channel_coefficients(measurement_ids.size)
    return TemperatureSensorDriftModel(
        measurement_ids=measurement_ids,
        coordinates=coordinates,
        normalised_coordinates=rbar,
        coordinate_reference_centre=centre,
        coordinate_reference_half_range=half_range,
        source_scale=source_scale,
        condition_u=condition_u,
        condition_g=condition_g,
        a=a,
        b=b,
        c=c,
        q=q,
        measurement_layout=str(measurement_layout),
        measurement_id_path=str(measurement_path),
        coordinate_reference_layout=str(coordinate_reference_layout),
        coordinate_reference_id_path=str(reference_path),
        source_health_path=str(Path(source_health_path).resolve()),
    )


def save_transformed_dataset(
    output_dir: str | Path,
    transformed: np.ndarray,
    diagnostics: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Save transformed ``V``, full numeric diagnostics and JSON configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset": output_dir / "preprocessed_data_raw.npz",
        "diagnostics": output_dir / "temperature_sensor_drift_diagnostics.npz",
        "config": output_dir / "temperature_sensor_drift_config.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: " + ", ".join(map(str, existing))
        )
    np.savez_compressed(paths["dataset"], V=np.asarray(transformed, dtype=np.float32))
    np.savez_compressed(
        paths["diagnostics"],
        **{key: np.asarray(value) for key, value in diagnostics.items()},
    )
    paths["config"].write_text(
        json.dumps(dict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return paths


def transform_dataset(
    values: np.ndarray,
    *,
    measurement_layout: str | Path = "original",
    temperature_span_c: float = MAIN_TEMPERATURE_SPAN_C,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Convenience wrapper for model construction, transformation and saving."""
    model = build_temperature_sensor_drift_model(measurement_layout)
    transformed, diagnostics, config = model.transform(values, temperature_span_c)
    if output_dir is not None:
        save_transformed_dataset(
            output_dir, transformed, diagnostics, config, overwrite=overwrite
        )
    return transformed, diagnostics, config


def _self_test() -> None:
    """Small synthetic test; it neither reads project data nor starts training."""
    n_channels = 7
    coordinates = np.column_stack(
        (
            np.linspace(-10.0, 10.0, n_channels),
            np.linspace(2.0, 5.0, n_channels),
            np.linspace(-3.0, 4.0, n_channels),
        )
    )
    rbar, centre, half_range = normalise_measurement_coordinates(coordinates)
    u, g = make_temperature_conditions()
    a, b, c, q = make_channel_coefficients(n_channels)
    model = TemperatureSensorDriftModel(
        measurement_ids=np.arange(1, n_channels + 1),
        coordinates=coordinates,
        normalised_coordinates=rbar,
        coordinate_reference_centre=centre,
        coordinate_reference_half_range=half_range,
        source_scale=np.linspace(0.5, 1.1, n_channels),
        condition_u=u,
        condition_g=g,
        a=a,
        b=b,
        c=c,
        q=q,
        measurement_layout="synthetic",
        measurement_id_path="synthetic",
        coordinate_reference_layout="synthetic",
        coordinate_reference_id_path="synthetic",
        source_health_path="synthetic healthy rows only",
    )
    rng = np.random.RandomState(7)
    x = rng.standard_normal((137, n_channels))
    y0, d0, _ = model.transform(x, 0.0)
    y1, d1, _ = model.transform(x, MAIN_TEMPERATURE_SPAN_C)
    y2, d2, _ = model.transform(x, MAIN_TEMPERATURE_SPAN_C)

    np.testing.assert_allclose(y0, x.astype(np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(d1["condition_index"][:100], np.arange(100))
    np.testing.assert_array_equal(d1["condition_index"][100:], np.arange(37))
    np.testing.assert_allclose(y1, y2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(d1["tau"], d2["tau"], rtol=0.0, atol=0.0)
    if np.allclose(y1, x):
        raise AssertionError("Non-zero temperature span did not alter synthetic data")
    if np.max(np.abs(d0["tau"])) > 1.0 + 1e-12:
        raise AssertionError("tau left its designed [-1, 1] range")
    if np.max(np.abs(d0["theta"])) > 1e-12:
        raise AssertionError("zero temperature span did not produce theta=0")
    print("temperature-conditioned sensor response self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a small synthetic reproducibility test; no project data or training",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return
    _self_test()


if __name__ == "__main__":
    main()
