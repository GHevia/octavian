from __future__ import annotations

import csv
from datetime import datetime, timezone

import numpy as np
import pytest
import spiceypy as spice

from octavian import Ephemeris, Solution
from octavian.coordinates import ric
from octavian.data.ephemeris import epoch_to_datetime_utc, epoch_to_et
from octavian.solvers.preconfigured import RendezvousResult


def _ephemeris() -> Ephemeris:
    times_s = np.asarray([0.0, 60.0, 120.0, 180.0])
    states = np.asarray(
        [
            [7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0],
            [6_985_000.0, 449_700.0, 0.0, -500.0, 7_484.0, 0.0],
            [6_940_000.0, 897_600.0, 0.0, -997.0, 7_436.0, 0.0],
            [6_865_000.0, 1_341_000.0, 0.0, -1_488.0, 7_356.0, 0.0],
        ]
    )
    return Ephemeris(
        times_s=times_s,
        states_m_mps=states,
        epoch="2026-01-01T00:00:00Z",
        object_name="TEST SAT",
        object_id=-123_456,
    )


def test_ephemeris_validates_rows_and_time_order() -> None:
    with pytest.raises(ValueError, match="shape"):
        Ephemeris(
            times_s=np.asarray([0.0, 1.0]),
            states_m_mps=np.zeros((2, 5)),
            epoch="2026-01-01T00:00:00Z",
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        Ephemeris(
            times_s=np.asarray([0.0, 0.0]),
            states_m_mps=np.zeros((2, 6)),
            epoch="2026-01-01T00:00:00Z",
        )


def test_epoch_datetime_conversion_round_trips_fallback() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recovered = epoch_to_datetime_utc(epoch_to_et(epoch))
    assert recovered == epoch


def test_stk_writer_uses_si_units_and_metadata(tmp_path) -> None:
    output = _ephemeris().write(tmp_path / "trajectory.e")
    text = output.read_text(encoding="utf-8")

    assert text.startswith("stk.v.12.0")
    assert "NumberOfEphemerisPoints 4" in text
    assert "ScenarioEpoch 01 Jan 2026 00:00:00.000000" in text
    assert "CentralBody Earth" in text
    assert "CoordinateSystem J2000" in text
    assert "DistanceUnit Meters" in text
    assert "7.0000000000000000e+06" in text


def test_oem_writer_uses_utc_and_kilometers(tmp_path) -> None:
    output = _ephemeris().write(tmp_path / "trajectory.oem")
    text = output.read_text(encoding="utf-8")

    assert "CCSDS_OEM_VERS = 2.0" in text
    assert "OBJECT_NAME = TEST SAT" in text
    assert "OBJECT_ID = -123456" in text
    assert "CENTER_NAME = EARTH" in text
    assert "REF_FRAME = J2000" in text
    assert "START_TIME = 2026-01-01T00:00:00.000000Z" in text
    assert "STOP_TIME = 2026-01-01T00:03:00.000000Z" in text
    assert "7.0000000000000000e+03" in text


def test_csv_writer_is_machine_readable_and_refuses_overwrite(tmp_path) -> None:
    ephemeris = _ephemeris()
    output = ephemeris.write(tmp_path / "trajectory.csv")
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 4
    assert rows[1]["epoch_utc"] == "2026-01-01T00:01:00.000000Z"
    assert float(rows[1]["x_m"]) == pytest.approx(6_985_000.0)
    with pytest.raises(FileExistsError):
        ephemeris.write(output)
    assert ephemeris.write(output, overwrite=True) == output


def test_spice_bsp_round_trips_state_and_object_id(tmp_path) -> None:
    ephemeris = _ephemeris()
    output = ephemeris.write(tmp_path / "trajectory.bsp", interpolation_degree=3)

    assert list(spice.spkobj(str(output))) == [ephemeris.object_id]
    spice.furnsh(str(output))
    try:
        state_km_kmps, _ = spice.spkez(
            ephemeris.object_id,
            float(ephemeris.epochs_et[1]),
            "J2000",
            "NONE",
            ephemeris.center_id,
        )
    finally:
        spice.unload(str(output))

    np.testing.assert_allclose(
        np.asarray(state_km_kmps) * 1_000.0,
        ephemeris.states_m_mps[1],
        rtol=1.0e-12,
        atol=1.0e-9,
    )


def test_solution_auto_selects_absolute_deputy_for_relative_export(tmp_path) -> None:
    times = np.asarray([0.0, 60.0, 120.0])
    relative = np.column_stack([np.zeros((3, 6)), times])
    chief = relative.copy()
    chief[:, 0] = 7_000_000.0
    chief[:, 4] = 7_500.0
    deputy = chief.copy()
    deputy[:, 0] += 100.0
    result = RendezvousResult(
        converged=True,
        traj=relative,
        maneuvers=[],
        info={
            "frame": ric("chief").to_dict(),
            "central_body": "earth",
            "chief_trajectory_eci": chief.tolist(),
            "deputy_trajectory_eci": deputy.tolist(),
        },
    )
    solution = Solution(
        ok=True,
        result=result,
        info={"initial_epoch": "2026-01-01T00:00:00Z"},
    )

    ephemeris = solution.to_ephemeris()
    np.testing.assert_allclose(ephemeris.states_m_mps, deputy[:, 0:6])
    assert ephemeris.frame_name == "J2000"
    assert ephemeris.object_name == "DEPUTY"
    output = solution.export_ephemeris(tmp_path / "deputy.csv")
    assert output.is_file()


def test_solution_requires_explicit_epoch_when_mission_did_not_supply_one() -> None:
    trajectory = np.column_stack([np.zeros((2, 6)), [0.0, 1.0]])
    solution = Solution(
        ok=True,
        result=RendezvousResult(converged=True, traj=trajectory, maneuvers=[]),
    )

    with pytest.raises(ValueError, match="initial_epoch"):
        solution.to_ephemeris()


def test_writer_rejects_unknown_extension(tmp_path) -> None:
    with pytest.raises(ValueError, match="Cannot infer"):
        _ephemeris().write(tmp_path / "trajectory.dat")
