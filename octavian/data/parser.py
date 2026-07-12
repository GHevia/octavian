from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np
import spiceypy as spice

DEFAULT_BODIES: dict[int, str] = {
    # 399: "EARTH",
    301: "MOON",
    10: "SUN",
}

def build_reduced_de440_bsp(
    input_bsp: str | Path,
    output_bsp: str | Path,
    *,
    timestep_days: Sequence[float] | None = None,
    interpolation_degree: int = 7,
    frame: str = "J2000",
    center_id: int = 0,
    bodies: Mapping[int, str] = DEFAULT_BODIES,
    sample_chunk_size: int = 50_000,
    overwrite: bool = False,
) -> Path:
    """
    Create a reduced BSP containing sampled Earth, Moon, and Sun states.

    Each body is stored relative to the Solar System Barycenter, NAIF ID 0,
    by default. Because all three bodies share the same center, SPICE can
    still compute their positions relative to one another.

    The output uses SPK Type 12:
        - evenly spaced states
        - position and velocity
        - Hermite interpolation

    Parameters
    ----------
    input_bsp
        Path to the original DE440 BSP.
    output_bsp
        Path for the reduced BSP.
    timestep_days
        Requested maximum time between samples, in days.
    interpolation_degree
        Odd Hermite polynomial degree between 1 and 27.
        Degree 7 is a reasonable starting point.
    frame
        Inertial frame in which states will be stored.
    center_id
        NAIF ID of the common center. Zero is the Solar System Barycenter.
    bodies
        Mapping from NAIF body ID to a readable name.
    sample_chunk_size
        Number of epochs evaluated in each SpiceyPy call.
    overwrite
        Replace an existing output BSP.

    Notes
    -----
    The timestep is adjusted slightly for each coverage interval so that:
        1. samples remain exactly evenly spaced, as required by Type 12;
        2. both original coverage endpoints are retained;
        3. the actual timestep is never larger than timestep_days.
    """
    input_path = Path(input_bsp).expanduser().resolve()
    output_path = Path(output_bsp).expanduser().resolve()

    timestep_days = np.array([1.0, 10.0] if timestep_days is None else timestep_days)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input BSP does not exist: {input_path}")

    if input_path == output_path:
        raise ValueError("Input and output BSP paths must be different.")

    if timestep_days[0] <= 0:
        raise ValueError("timestep_days must be positive.")

    if not 1 <= interpolation_degree <= 27:
        raise ValueError("interpolation_degree must be between 1 and 27.")

    if interpolation_degree % 2 == 0:
        raise ValueError(
            "SPK Type 12 requires an odd interpolation degree."
        )

    if sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be at least 1.")

    if not bodies:
        raise ValueError("At least one body must be requested.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Pass overwrite=True to replace it."
            )
        output_path.unlink()

    requested_step_seconds = timestep_days * spice.spd()
    minimum_states = (interpolation_degree + 1) // 2

    handle: int | None = None
    source_loaded = False

    try:
        # Loading DE440 allows SPICE to combine its internal segments and
        # produce direct states relative to the Solar System Barycenter.
        spice.furnsh(str(input_path))
        source_loaded = True

        handle = spice.spkopn(
            str(output_path),
            "Sampled DE440 Earth Moon Sun",
            0,  # Characters reserved for comments.
        )
        jj = -1
        for body_id, body_name in bodies.items():
            jj+=1
            # spkcov returns all intervals in which the body has data.
            coverage = spice.spkcov(str(input_path), int(body_id))
            interval_count = spice.wncard(coverage)

            if interval_count == 0:
                raise ValueError(
                    f"No SPK coverage found for {body_name} "
                    f"(NAIF ID {body_id}) in {input_path}."
                )

            for interval_index in range(interval_count):
                first, last = spice.wnfetd(coverage, interval_index)
                duration = last - first

                if duration <= 0:
                    raise ValueError(
                        f"Degenerate coverage interval for {body_name}: "
                        f"{first} to {last}."
                    )

                # Choose a state count that keeps the actual step no greater
                # than the requested step while landing exactly on both ends.
                number_of_steps = max(
                    1,
                    math.ceil(duration / requested_step_seconds[jj]),
                )

                n_states = max(
                    minimum_states,
                    number_of_steps + 1,
                )

                actual_step = duration / (n_states - 1)

                epochs = (
                    first
                    + actual_step
                    * np.arange(n_states, dtype=np.float64)
                )

                states = np.empty((n_states, 6), dtype=np.float64)

                # Evaluate in chunks rather than making one enormous call.
                for i0 in range(0, n_states, sample_chunk_size):
                    i1 = min(i0 + sample_chunk_size, n_states)

                    chunk_states, _ = spice.spkezr(
                        str(int(body_id)),
                        epochs[i0:i1],
                        frame,
                        "NONE",
                        str(int(center_id)),
                    )

                    states[i0:i1] = np.asarray(
                        chunk_states,
                        dtype=np.float64,
                    ).reshape(-1, 6)

                # SPK segment IDs may contain at most 40 characters.
                segment_id = (
                    f"DE440 sampled {body_name} {interval_index}"
                )[:40]

                spice.spkw12(
                    handle,
                    int(body_id),
                    int(center_id),
                    frame,
                    first,
                    last,
                    segment_id,
                    interpolation_degree,
                    n_states,
                    states,
                    first,
                    actual_step,
                )

                print(
                    f"{body_name:5s}: "
                    f"interval {interval_index}, "
                    f"{n_states:,} states, "
                    f"step={actual_step / spice.spd():.9f} days"
                )

        spice.spkcls(handle)
        handle = None

        size_mib = output_path.stat().st_size / (1024**2)
        print(f"\nCreated: {output_path}")
        print(f"File size: {size_mib:.2f} MiB")

        return output_path

    except Exception:
        # Close and remove a partially written kernel.
        if handle is not None:
            with suppress(Exception):
                spice.spkcls(handle)

        if output_path.exists():
            output_path.unlink()

        raise

    finally:
        if source_loaded:
            spice.unload(str(input_path))

if __name__ == "__main__":
    build_reduced_de440_bsp(
    input_bsp="de440.bsp",
    output_bsp="sun_moon_scheduled.bsp",
    timestep_days=[1.0, 10],
    interpolation_degree=5,
    center_id=399,
    overwrite=True,
)
    
    import spiceypy as spice

    spice.furnsh("sun_moon_scheduled.bsp")

    et = 0.0

    moon_state, _ = spice.spkezr(
        "MOON",
        et,
        "J2000",
        "NONE",
        "EARTH",
    )

    sun_state, _ = spice.spkezr(
        "SUN",
        et,
        "J2000",
        "NONE",
        "EARTH",
    )

    print("Moon position [km]:", moon_state[:3])
    print("Moon velocity [km/s]:", moon_state[3:])

    print("Sun position [km]:", sun_state[:3])
    print("Sun velocity [km/s]:", sun_state[3:])

    spice.kclear()
