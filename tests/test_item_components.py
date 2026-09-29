import pytest

from rapidcadpy.cad_session import CadSession
from rapidcadpy.components import profiles
from rapidcadpy.components.item import item_angle_bracket


class _RecordingWorkplane:
    """Minimal workplane double that records ITEM section sketch loops."""

    def __init__(self) -> None:
        self.operations: list[tuple[str, tuple[float, ...]]] = []

    def move_to(self, x: float, y: float):
        self.operations.append(("move_to", (x, y)))
        return self

    def line_to(self, x: float, y: float):
        self.operations.append(("line_to", (x, y)))
        return self

    def circle(self, radius: float):
        self.operations.append(("circle", (radius,)))
        return self

    def close(self):
        self.operations.append(("close", ()))
        return self


def test_session_exposes_item_angle_bracket_catalog():
    session = CadSession()

    listed = session.list_item_angle_brackets()
    loaded = session.get_item_angle_bracket("ITEM5_20X20_ANGLE_BRACKET")

    assert listed["ok"] is True
    assert listed["brackets"] == [
        {
            "name": "ITEM5_20X20_ANGLE_BRACKET",
            "part_number": "0.0.425.03",
            "leg_length": 20.0,
            "width": 20.0,
            "height": 20.0,
            "mounting_hole_diameter": 5.3,
            "mounting_hole_count": 2,
            "material": "die_cast_zinc",
            "mass_g": 14.0,
            "joint_angle_degrees": 90.0,
        }
    ]
    assert loaded["ok"] is True
    assert loaded["bracket"] == listed["brackets"][0]


def test_item_section_closes_outer_and_bore_loops_separately():
    workplane = _RecordingWorkplane()
    section = profiles.item("ITEM5_20X20")

    returned_workplane = section.sketch(workplane)

    assert returned_workplane is workplane
    assert [name for name, _ in workplane.operations].count("close") == 2
    assert workplane.operations[-3:] == [
        ("move_to", (0.0, 0.0)),
        ("circle", (section.core_hole_diameter / 2.0,)),
        ("close", ()),
    ]


@pytest.fixture
def freecad_app():
    try:
        from rapidcadpy.integrations.freecad import FreeCADApp

        return FreeCADApp()
    except ImportError as exc:
        pytest.skip(f"FreeCAD is not importable in this Python runtime: {exc}")


@pytest.fixture
def freecad_ui_app():
    """Launch a bridge-enabled FreeCAD UI and return its RapidCADPy app."""
    from rapidcadpy.integrations.freecad import FreeCADApp
    from rapidcadpy.integrations.freecad.gui_connection import (
        FreeCADGuiConnection,
        discover_freecad_installation,
    )

    if discover_freecad_installation() is None:
        pytest.skip("FreeCAD GUI executable is not installed")

    connection = FreeCADGuiConnection.launch()
    app = FreeCADApp("attach", instance_id=connection.instance_id)
    yield app

    # FreeCADGuiConnection.close() deliberately leaves the launched desktop
    # application open so the generated assembly remains available to inspect.
    connection.close()


@pytest.mark.integration
@pytest.mark.serial
def test_item_profiles_form_square_frame_with_angle_bracket_connections(
    freecad_app,
    freecad_ui_app,
    tmp_path,
):
    """Build four ITEM members as a square and assign the corner hardware.

    ``ItemAngleBracket`` is currently a catalog component, not a geometric
    builder. This integration test therefore validates the four native ITEM
    profile extrusions and the standardized bracket selected for every corner;
    it does not imply that the bracket is represented by a CAD solid.
    """
    section = profiles.item("ITEM5_20X20")
    bracket = item_angle_bracket("ITEM5_20X20_ANGLE_BRACKET")
    side_length = 200.0

    members = {
        "left": section.sketch(
            freecad_app.work_plane(
                origin=(0.0, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
            )
        ).extrude(side_length),
        "right": section.sketch(
            freecad_app.work_plane(
                origin=(side_length, 0.0, 0.0),
                normal=(0.0, 0.0, 1.0),
            )
        ).extrude(side_length),
        "bottom": section.sketch(
            freecad_app.work_plane(
                origin=(0.0, 0.0, 0.0),
                normal=(1.0, 0.0, 0.0),
            )
        ).extrude(side_length),
        "top": section.sketch(
            freecad_app.work_plane(
                origin=(0.0, 0.0, side_length),
                normal=(1.0, 0.0, 0.0),
            )
        ).extrude(side_length),
    }
    corner_connections = {
        "bottom_left": ("left", "bottom"),
        "bottom_right": ("right", "bottom"),
        "top_left": ("left", "top"),
        "top_right": ("right", "top"),
    }

    assert all(
        member is not None and member.obj.isValid() for member in members.values()
    )
    assert members["left"].obj.BoundBox.ZLength == pytest.approx(side_length)
    assert members["right"].obj.BoundBox.ZLength == pytest.approx(side_length)
    assert members["bottom"].obj.BoundBox.XLength == pytest.approx(side_length)
    assert members["top"].obj.BoundBox.XLength == pytest.approx(side_length)
    assert all(member.volume() > 0.0 for member in members.values())
    assert all(
        member.volume() < section.width * section.height * side_length
        for member in members.values()
    )
    assert all(
        member.feature.native_handle.LengthFwd.Value == pytest.approx(side_length)
        for member in members.values()
    )
    assert all(
        member.feature.native_handle.LengthRev.Value == pytest.approx(0.0)
        for member in members.values()
    )
    assert set(corner_connections) == {
        "bottom_left",
        "bottom_right",
        "top_left",
        "top_right",
    }
    assert all(
        set(connection).issubset(members) for connection in corner_connections.values()
    )
    assert bracket.part_number == "0.0.425.03"
    assert bracket.joint_angle_degrees == pytest.approx(90.0)
    assert bracket.mounting_hole_count == 2

    output_path = tmp_path / "item_square_frame.FCStd"
    freecad_app.to_fcstd(str(output_path), shapes=list(members.values()))
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    opened = freecad_ui_app.call("open_document", path=str(output_path))

    assert opened["ok"] is True, opened.get("error")
    assert opened["object_count"] >= 4
    assert opened["document"]["file_name"] == str(output_path)
