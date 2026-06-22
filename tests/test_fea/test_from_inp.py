import pytest
import pathlib
import textwrap
import numpy as np

from rapidcadpy.fea.load_case.load_case import LoadCase
from rapidcadpy.fea.load_case.abaqus_inp_load_case import AbaqusInpLoadCase
from rapidcadpy.fea.load_case.freecad_inp_load_case import LoadCaseFromFreeCadInp
from rapidcadpy.fea.boundary_conditions import (
    AccelerationLoad,
    FixedConstraint,
    PointLoad,
)


def _has_opengl() -> bool:
    """Return True only when a usable OpenGL context can be created.

    Headless CI / RDP sessions without a GPU will fail to initialise a
    VTK render window, so any test that calls ``fea.show()`` should be
    guarded by this check.
    """
    try:
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        import vtkmodules.vtkRenderingCore as rc

        rw = rc.vtkRenderWindow()
        rw.SetOffScreenRendering(0)
        rw.Initialize()
        return rw.GetSize()[0] > 0
    except Exception:
        return False


class TestFromInp:
    @pytest.fixture(params=["fea_test_4", "fea_test_1_abaqus", "fea_test_3_2_abaqus"])
    def get_inp_file(self, request) -> dict:
        """Parametrized fixture: runs tests for both FEA input files."""
        inp_file = pathlib.Path(__file__).parent.parent / "test_files" / "fea"

        cases = {
            "00004968": {
                "path": inp_file / "00004968.inp",
                "problem_id": "00004968",
                "description_contains": "00004968.inp",
                "bounds": {
                    "x_min": -0.235537,
                    "x_max": 0.235537,
                    "y_min": -0.661157,
                    "y_max": 0.0,
                    "z_min": -0.5,
                    "z_max": 0.5,
                },
                "bc_dofs": (True, True, True),
                "load_force": (0.0, 0.0, -125.0),
                "load_magnitude": 125.0,
                "node_sets_count_min": 2,
                "abaqus_element_type": None,
                "mesh_nodes_shape": (24, 3),
                "mesh_elements_shape": (36, 4),
                "mesh_element_type": "tet4",
            },
            "fea_test_4": {
                "path": inp_file / "fea_test_4_freecad.inp",
                "problem_id": "FEA_TEST_4_FREECAD",
                "description_contains": "fea_test_4_freecad.inp",
                "bounds": {
                    "x_min": -52.477569,
                    "x_max": 47.522431,
                    "y_min": -25.0,
                    "y_max": 0.0,
                    "z_min": 0.0,
                    "z_max": 27.127161,
                },
                "bc_dofs": (True, True, True),
                "load_force": (0.0, 0.0, -100.0),
                "load_magnitude": 100.0,
                "node_sets_count_min": 1,
                "abaqus_element_type": None,
                "mesh_nodes_shape": (1102, 3),
                "mesh_elements_shape": (519, 10),
                "mesh_element_type": "tet10",
            },
            "fea_test_1": {
                "path": inp_file / "fea_test_1.inp",
                "problem_id": "FEA_TEST_1",
                "description_contains": "fea_test_1.inp",
                "bounds": {
                    "x_min": 0.0,
                    "x_max": 30.0,
                    "y_min": 0.0,
                    "y_max": 30.0,
                    "z_min": 0.0,
                    "z_max": 3.0,
                },
                "bc_dofs": (True, True, True),
                "load_force": (0.0, 0.0, -100.0),
                "load_magnitude": 100.0,
                "node_sets_count_min": 1,
                "abaqus_element_type": None,
                "mesh_nodes_shape": (1460, 3),
                "mesh_elements_shape": (701, 10),
                "mesh_element_type": "tet10",
            },
            "fea_test_1_abaqus": {
                "path": inp_file / "fea_test_1_abaqus.inp",
                "problem_id": "FEA_TEST_1_ABAQUS",
                "description_contains": "fea_test_1_abaqus.inp",
                # Only 2 inline nodes (89999 at origin, 99999 at y=1).
                # All structural nodes come from an external INPUT= file that
                # is not present in the repo.
                "bounds": {
                    "x_min": 0.0,
                    "x_max": 0.0,
                    "y_min": 0.0,
                    "y_max": 1.0,
                    "z_min": 0.0,
                    "z_max": 0.0,
                },
                # BCs and loads reference external-only node IDs so they
                # cannot be resolved against the inline node table.
                "bc_dofs": None,
                "load_force": None,
                "load_magnitude": None,
                "node_sets_count_min": 18,
                "abaqus_element_type": None,
                # 480 C3D20R elements (two inline blocks) × 20 nodes each
                "mesh_nodes_shape": (2, 3),
                "mesh_elements_shape": (480, 20),
                "mesh_element_type": "hex20",
                # Flag: BCs/loads/selectors cannot be populated for this file
                "external_nodes_only": True,
            },
            "fea_test_3_2_abaqus": {
                "path": inp_file / "fea_test_3_2_abaqus.inp",
                "problem_id": "FEA_TEST_3_2_ABAQUS",
                "description_contains": "fea_test_3_2_abaqus.inp",
                # Entire mesh lives in *INCLUDE'd Boogie_bracket_V03.inp.
                # Tests are auto-skipped when that file is absent.
                # When present, the parser must produce a non-empty mesh,
                # proving that *INCLUDE expansion actually worked.
                "required_include_files": ["fea_test_3_1_abaqus.inp"],
                "bounds": {
                    "x_min": -896.999995,
                    "x_max": 4385.0,
                    "y_min": -552.500008,
                    "y_max": 333.537,
                    "z_min": -538.739199,
                    "z_max": 182.103005,
                },
                # boundary_nodes, 1, 6, 0  → DOFs 1-6 all locked
                "bc_dofs": (True, True, True),
                # Multi-axis CLOADs in x/y/z — per-component check skipped
                "load_force": None,
                "load_magnitude": None,
                # boundary_nodes inline + 4 more from included file
                "node_sets_count_min": 5,
                "abaqus_element_type": None,
                # 824 456 nodes, 502 901 C3D10HS elements → tet10
                "mesh_nodes_shape": (824456, 3),
                "mesh_elements_shape": (502901, 10),
                "mesh_element_type": "tet10",
            },
        }

        return cases[request.param]

    @pytest.fixture
    def load_case(self, get_inp_file) -> LoadCase:
        """Parse the sample .inp file once and reuse across tests.

        Routes to the native Abaqus parser when the problem_id contains
        'abaqus' (case-insensitive), otherwise uses the FreeCAD/meshio path.

        Skips automatically when any file listed in ``required_include_files``
        is absent — those tests only make sense once the *INCLUDE'd mesh is
        available.
        """
        base_dir = get_inp_file["path"].parent
        for inc_name in get_inp_file.get("required_include_files", []):
            inc_path = base_dir / inc_name
            if not inc_path.exists():
                pytest.skip(
                    f"Required *INCLUDE file not available: {inc_name} "
                    f"(expected at {inc_path})"
                )

        path = str(get_inp_file["path"])
        if "abaqus" in get_inp_file["problem_id"].lower():
            return AbaqusInpLoadCase.from_inp(path)
        return LoadCaseFromFreeCadInp.from_inp(path)

    # ------------------------------------------------------------------
    # Basic return type / structure
    # ------------------------------------------------------------------

    def test_returns_load_case_instance(self, load_case):
        assert isinstance(load_case, LoadCase)

    def test_problem_id_is_set(self, load_case, get_inp_file):
        assert load_case.problem_id == get_inp_file["problem_id"]

    def test_description_contains_filename(self, load_case, get_inp_file):
        assert get_inp_file["description_contains"] in load_case.description

    # ------------------------------------------------------------------
    # Nodes / bounds
    # ------------------------------------------------------------------

    def test_bounds_are_populated(self, load_case):
        assert load_case.bounds is not None
        for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            assert key in load_case.bounds

    def test_bounds_min_less_than_max(self, load_case):
        b = load_case.bounds
        assert b["x_min"] <= b["x_max"]
        assert b["y_min"] <= b["y_max"]
        assert b["z_min"] <= b["z_max"]

    def test_bounds_values_match_node_extents(self, load_case, get_inp_file):
        """Bounds should span the node coordinate range in the file."""
        b = load_case.bounds
        expected = get_inp_file["bounds"]
        if expected is None:
            # Exact values unknown (come from *INCLUDE'd file); just assert
            # that at least one axis has a non-zero span, proving the mesh
            # was actually loaded.
            assert any(
                b[mx] > b[mn]
                for mn, mx in [
                    ("x_min", "x_max"),
                    ("y_min", "y_max"),
                    ("z_min", "z_max"),
                ]
            ), "Bounds are all-zero — *INCLUDE expansion likely failed"
            return
        for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            assert b[key] == pytest.approx(expected[key], abs=1e-5)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def test_boundary_conditions_not_empty(self, load_case, get_inp_file):
        if get_inp_file.get("external_nodes_only"):
            pytest.skip("BCs reference external-only node IDs, cannot be resolved")
        assert len(load_case.boundary_conditions) > 0

    def test_boundary_condition_type_is_fixed_constraint(self, load_case, get_inp_file):
        if get_inp_file.get("external_nodes_only"):
            pytest.skip("BCs reference external-only node IDs, cannot be resolved")
        for bc in load_case.boundary_conditions:
            assert isinstance(bc, FixedConstraint)

    def test_fixed_constraint_locks_all_three_dofs(self, load_case, get_inp_file):
        if get_inp_file.get("bc_dofs") is None:
            pytest.skip("bc_dofs unknown for this test case")
        bc = load_case.boundary_conditions[0]
        assert bc.dofs == get_inp_file["bc_dofs"]

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def test_loads_not_empty(self, load_case, get_inp_file):
        if get_inp_file.get("external_nodes_only"):
            pytest.skip("CLOADs reference external-only node IDs, cannot be resolved")
        assert len(load_case.loads) > 0

    def test_load_type_is_point_load(self, load_case, get_inp_file):
        if get_inp_file.get("external_nodes_only"):
            pytest.skip("CLOADs reference external-only node IDs, cannot be resolved")
        for load in load_case.loads:
            assert isinstance(load, PointLoad)

    def test_load_force_is_in_z_direction(self, load_case, get_inp_file):
        if get_inp_file.get("load_force") is None:
            pytest.skip("load_force not specified for this test case")
        load = load_case.loads[0]
        expected_fx, expected_fy, expected_fz = get_inp_file["load_force"]
        fx, fy, fz = load.force
        assert fx == pytest.approx(expected_fx, abs=1e-5)
        assert fy == pytest.approx(expected_fy, abs=1e-5)
        assert fz == pytest.approx(expected_fz, abs=1e-5)

    def test_load_magnitude_newtons(self, load_case, get_inp_file):
        if get_inp_file.get("load_magnitude") is None:
            pytest.skip("load_magnitude not specified for this test case")
        assert load_case.loads[0].magnitude_newtons == pytest.approx(
            get_inp_file["load_magnitude"], abs=1e-5
        )

    # ------------------------------------------------------------------
    # Meta / selectors
    # ------------------------------------------------------------------

    def test_meta_is_populated(self, load_case):
        assert load_case.meta is not None

    def test_meta_contains_node_sets_count(self, load_case, get_inp_file):
        assert load_case.meta["node_sets_count"] >= get_inp_file["node_sets_count_min"]

    def test_selectors_populated(self, load_case, get_inp_file):
        assert hasattr(load_case, "selectors")
        if get_inp_file.get("external_nodes_only"):
            pytest.skip("NSETs reference external-only node IDs, no selectors built")
        assert len(load_case.selectors) > 0

    def test_abaqus_element_type_detected(self, load_case, get_inp_file):
        assert (
            load_case.meta.get("abaqus_element_type")
            == get_inp_file["abaqus_element_type"]
        )

    # ------------------------------------------------------------------
    # Mesh data
    # ------------------------------------------------------------------

    def test_mesh_nodes_is_ndarray(self, load_case):
        assert isinstance(load_case.mesh_nodes, np.ndarray)

    def test_mesh_nodes_shape(self, load_case, get_inp_file):
        expected = get_inp_file["mesh_nodes_shape"]
        if expected is None:
            assert (
                load_case.mesh_nodes.shape[0] > 0
            ), "mesh_nodes is empty — *INCLUDE expansion likely failed"
            assert load_case.mesh_nodes.shape[1] == 3
            return
        assert load_case.mesh_nodes.shape == expected

    def test_mesh_elements_is_ndarray(self, load_case):
        assert isinstance(load_case.mesh_elements, np.ndarray)

    def test_mesh_elements_shape(self, load_case, get_inp_file):
        expected = get_inp_file["mesh_elements_shape"]
        if expected is None:
            assert (
                load_case.mesh_elements.shape[0] > 0
            ), "mesh_elements is empty — *INCLUDE expansion likely failed"
            return
        assert load_case.mesh_elements.shape == expected

    def test_mesh_element_type_is_tet4(self, load_case, get_inp_file):
        expected = get_inp_file["mesh_element_type"]
        if expected is None:
            pytest.skip("mesh_element_type unknown for this test case")
        assert load_case.mesh_element_type == expected

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            LoadCase.from_inp("/nonexistent/path/to/file.inp")

    # ---
    # Visualization
    ###
    def test_plot_fea(self, load_case, get_inp_file):
        # Use "mesh" mode when BCs aren't meaningful (external_nodes_only files
        # have all element connectivity collapsed to a two-node inline stub).
        mode = "mesh" if get_inp_file.get("bc_dofs") is None else "conditions"
        fea = load_case.get_fea_analyzer()
        fea.show(display=mode, filename="test_plot_fea_conditions.png")


class TestFromInpWithFea:
    @pytest.fixture
    def get_inp_file(self):
        inp_file = pathlib.Path(__file__).parent.parent / "test_files"
        return inp_file / "00004968.inp"

    @pytest.fixture
    def get_inp_file_bracket(self):
        return (
            "/Users/elias.berger/Documents/agentic_cad_system/data/abaqus/l_bracket.inp"
        )

    @pytest.fixture
    def load_case(self, get_inp_file):
        """Parse the sample .inp file once and reuse across tests."""
        return LoadCase.from_inp(str(get_inp_file))

    @pytest.fixture
    def load_case_bracket(self, get_inp_file_bracket):
        """Parse the bracket .inp file once and reuse across tests."""
        return LoadCase.from_inp(str(get_inp_file_bracket))

    def test_full_fea_solve(self, load_case_bracket):
        """Test that we can run a full FEA solve on the loaded case."""
        # This is more of an integration test since it depends on the meshing and solver.
        # But it's important to verify that from_inp produces a load case that can be solved end-to-end.
        mesh_size = load_case_bracket.calc_mesh_size(num_nodes=100)
        fea = load_case_bracket.get_fea_analyzer("gmsh", mesh_size=mesh_size)
        fea.show("conditions")


class TestToInp:
    def test_to_inp_round_trip_with_existing_mesh(self, tmp_path):
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

        lc = LoadCase(problem_id="ROUNDTRIP", description="round-trip test")
        lc.mesh_nodes = nodes
        lc.mesh_elements = elements
        lc.mesh_element_type = "tet4"
        lc.bounds = {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "z_min": 0.0,
            "z_max": 1.0,
        }
        lc.add_constraint(FixedConstraint(location=(0.0, 0.0, 0.0)))
        lc.add_load(PointLoad(point=(1.0, 0.0, 0.0), force=(0.0, -500.0, 0.0)))

        out_path = tmp_path / "roundtrip.inp"
        written = lc.to_inp(out_path)

        assert written.exists()

        parsed = LoadCase.from_inp(str(written))
        assert parsed.mesh_nodes is not None
        assert parsed.mesh_elements is not None
        assert parsed.mesh_nodes.shape == (4, 3)
        assert parsed.mesh_elements.shape == (1, 4)
        assert len(parsed.boundary_conditions) >= 1
        assert len(parsed.loads) >= 1


class TestToInpFromInp:
    @staticmethod
    def _assert_location_close(original_loc, round_trip_loc, tol=1e-5):
        # dict box/point locations
        if isinstance(original_loc, dict) and isinstance(round_trip_loc, dict):
            keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
            point_keys = ["x", "y", "z"]

            if all(k in original_loc for k in point_keys) and all(
                k in round_trip_loc for k in point_keys
            ):
                for key in point_keys:
                    assert round_trip_loc[key] == pytest.approx(
                        original_loc[key], abs=tol
                    )
                return

            # Box extents are parser-generated heuristics and may differ after
            # INP round-trip while preserving physical location. Compare centroids
            # (or explicit x/y/z if present) rather than exact min/max planes.
            if all(k in original_loc for k in keys) and all(
                k in round_trip_loc for k in keys
            ):
                orig_c = (
                    0.5 * (original_loc["x_min"] + original_loc["x_max"]),
                    0.5 * (original_loc["y_min"] + original_loc["y_max"]),
                    0.5 * (original_loc["z_min"] + original_loc["z_max"]),
                )
                rt_c = (
                    0.5 * (round_trip_loc["x_min"] + round_trip_loc["x_max"]),
                    0.5 * (round_trip_loc["y_min"] + round_trip_loc["y_max"]),
                    0.5 * (round_trip_loc["z_min"] + round_trip_loc["z_max"]),
                )
                for a, b in zip(rt_c, orig_c):
                    assert a == pytest.approx(b, abs=max(tol, 1e-3))
                return

        # tuple/list point locations
        if isinstance(original_loc, (tuple, list)) and isinstance(
            round_trip_loc, (tuple, list)
        ):
            assert len(original_loc) >= 3
            assert len(round_trip_loc) >= 3
            for i in range(3):
                assert round_trip_loc[i] == pytest.approx(original_loc[i], abs=tol)
            return

        assert round_trip_loc == original_loc

    def test_inp_to_inp_round_trip(self, get_inp_file, tmp_path):
        if get_inp_file.get("external_nodes_only"):
            pytest.skip(
                "Round-trip test requires resolvable BCs/loads; "
                "this file uses external-only nodes"
            )
        original = LoadCase.from_inp(str(get_inp_file["path"]))
        out_path = tmp_path / "roundtrip.inp"
        original.to_inp(out_path)

        assert out_path.exists()

        round_trip = LoadCase.from_inp(str(out_path))
        assert round_trip.mesh_nodes is not None
        assert round_trip.mesh_elements is not None
        assert original.mesh_nodes is not None
        assert original.mesh_elements is not None

        orig_nodes = original.mesh_nodes
        orig_elems = original.mesh_elements
        rt_nodes = round_trip.mesh_nodes
        rt_elems = round_trip.mesh_elements

        # Mesh-level equivalence
        assert rt_nodes.shape == orig_nodes.shape
        assert rt_elems.shape == orig_elems.shape
        np.testing.assert_allclose(rt_nodes, orig_nodes, atol=1e-5)
        np.testing.assert_array_equal(rt_elems, orig_elems)
        assert round_trip.mesh_element_type == original.mesh_element_type

        # Bounds equivalence
        assert round_trip.bounds is not None and original.bounds is not None
        for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            assert round_trip.bounds[key] == pytest.approx(
                original.bounds[key], abs=1e-5
            )

        # Constraint equivalence
        assert len(round_trip.boundary_conditions) == len(original.boundary_conditions)
        assert len(round_trip.boundary_conditions) > 0
        for bc_orig, bc_rt in zip(
            original.boundary_conditions, round_trip.boundary_conditions
        ):
            assert type(bc_rt) is type(bc_orig)
            assert getattr(bc_rt, "dofs", None) == getattr(bc_orig, "dofs", None)
            assert getattr(bc_rt, "tolerance", None) == pytest.approx(
                getattr(bc_orig, "tolerance", None), abs=1e-8
            )
            self._assert_location_close(
                getattr(bc_orig, "location", None),
                getattr(bc_rt, "location", None),
            )

        # Load equivalence
        assert len(round_trip.loads) == len(original.loads)
        assert len(round_trip.loads) > 0

        total_force_orig = np.zeros(3, dtype=np.float64)
        total_force_rt = np.zeros(3, dtype=np.float64)

        for load_orig, load_rt in zip(original.loads, round_trip.loads):
            assert type(load_rt) is type(load_orig)

            f_orig = getattr(load_orig, "force", (0.0, 0.0, 0.0))
            f_rt = getattr(load_rt, "force", (0.0, 0.0, 0.0))
            if isinstance(f_orig, (int, float)):
                force_orig = np.array([0.0, 0.0, float(f_orig)], dtype=np.float64)
            else:
                force_orig = np.array(f_orig, dtype=np.float64)
            if isinstance(f_rt, (int, float)):
                force_rt = np.array([0.0, 0.0, float(f_rt)], dtype=np.float64)
            else:
                force_rt = np.array(f_rt, dtype=np.float64)
            np.testing.assert_allclose(force_rt, force_orig, atol=1e-6)

            total_force_orig += force_orig
            total_force_rt += force_rt

            self._assert_location_close(
                getattr(load_orig, "point", None),
                getattr(load_rt, "point", None),
            )

            assert getattr(load_rt, "direction", None) == getattr(
                load_orig, "direction", None
            )
            assert getattr(load_rt, "magnitude_newtons", None) == pytest.approx(
                getattr(load_orig, "magnitude_newtons", None), abs=1e-8
            )

            sr_orig = getattr(load_orig, "search_radius", None)
            sr_rt = getattr(load_rt, "search_radius", None)
            if sr_orig is None or sr_rt is None:
                assert sr_orig is None and sr_rt is None
            else:
                np.testing.assert_allclose(
                    np.array(sr_rt), np.array(sr_orig), atol=1e-6
                )

        np.testing.assert_allclose(total_force_rt, total_force_orig, atol=1e-6)


class TestGravityLoadFromInp:
    """
    Tests for parsing a direct Abaqus export that contains a *DLOAD, GRAV
    body-force gravity load and *BOUNDARY displacement constraints.

    Source file: tests/test_files/fea/fea_test_abaqus_gravity_load.inp
    """

    INP_FILENAME = "fea_test_abaqus_gravity_load.inp"

    # Expected values derived from a verified parse run
    EXPECTED = {
        "problem_id": "FEA_TEST_ABAQUS_GRAVITY_LOAD",
        "mesh_nodes_shape": (237, 3),
        "mesh_elements_shape": (688, 4),
        "mesh_element_type": "tet4",
        "bounds": {
            "x_min": -9.55653095,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 25.3999996,
            "z_min": -9.52053547,
            "z_max": 0.0,
        },
        "n_boundary_conditions": 4,
        "n_loads": 1,
        # Gravity load specifics — taken directly from the *DLOAD section
        "gravity_magnitude": 9.71,
        "gravity_direction": (0.0, 0.0, -1.0),
        "gravity_element_set": "EALL",
    }

    @pytest.fixture
    def inp_path(self):
        return (
            pathlib.Path(__file__).parent.parent
            / "test_files"
            / "fea"
            / self.INP_FILENAME
        )

    @pytest.fixture
    def load_case(self, inp_path):
        return LoadCase.from_inp(str(inp_path))

    @pytest.fixture
    def gravity_load(self, load_case):
        """Return the first AccelerationLoad in the load case."""
        for load in load_case.loads:
            if isinstance(load, AccelerationLoad):
                return load
        pytest.fail("No AccelerationLoad found in parsed load case")

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_returns_load_case_instance(self, load_case):
        assert isinstance(load_case, LoadCase)

    def test_problem_id(self, load_case):
        assert load_case.problem_id == self.EXPECTED["problem_id"]

    def test_description_contains_filename(self, load_case):
        assert self.INP_FILENAME in load_case.description

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------

    def test_mesh_nodes_shape(self, load_case):
        assert load_case.mesh_nodes.shape == self.EXPECTED["mesh_nodes_shape"]

    def test_mesh_elements_shape(self, load_case):
        assert load_case.mesh_elements.shape == self.EXPECTED["mesh_elements_shape"]

    def test_mesh_element_type(self, load_case):
        assert load_case.mesh_element_type == self.EXPECTED["mesh_element_type"]

    # ------------------------------------------------------------------
    # Bounds
    # ------------------------------------------------------------------

    def test_bounds_populated(self, load_case):
        assert load_case.bounds is not None

    def test_bounds_values(self, load_case):
        b = load_case.bounds
        exp = self.EXPECTED["bounds"]
        for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            assert b[key] == pytest.approx(exp[key], abs=1e-5)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def test_boundary_conditions_not_empty(self, load_case):
        assert len(load_case.boundary_conditions) > 0

    def test_boundary_condition_count(self, load_case):
        assert (
            len(load_case.boundary_conditions) == self.EXPECTED["n_boundary_conditions"]
        )

    def test_all_bcs_are_fixed_constraints(self, load_case):
        for bc in load_case.boundary_conditions:
            assert isinstance(bc, FixedConstraint)

    def test_fixed_constraints_lock_all_dofs(self, load_case):
        for bc in load_case.boundary_conditions:
            assert bc.dofs == (True, True, True)

    # ------------------------------------------------------------------
    # Loads — presence
    # ------------------------------------------------------------------

    def test_exactly_one_load(self, load_case):
        assert len(load_case.loads) == self.EXPECTED["n_loads"]

    def test_load_is_acceleration_load(self, gravity_load):
        assert isinstance(gravity_load, AccelerationLoad)

    # ------------------------------------------------------------------
    # Gravity load attributes
    # ------------------------------------------------------------------

    def test_gravity_load_type(self, gravity_load):
        assert gravity_load.load_type == AccelerationLoad.GRAVITY

    def test_gravity_magnitude(self, gravity_load):
        assert gravity_load.magnitude == pytest.approx(
            self.EXPECTED["gravity_magnitude"], abs=1e-6
        )

    def test_gravity_direction_z_down(self, gravity_load):
        dx, dy, dz = gravity_load.direction
        assert dx == pytest.approx(self.EXPECTED["gravity_direction"][0], abs=1e-6)
        assert dy == pytest.approx(self.EXPECTED["gravity_direction"][1], abs=1e-6)
        assert dz == pytest.approx(self.EXPECTED["gravity_direction"][2], abs=1e-6)

    def test_gravity_element_set(self, gravity_load):
        assert gravity_load.element_set == self.EXPECTED["gravity_element_set"]

    def test_gravity_density_is_none_when_not_in_inp(self, gravity_load):
        """Density is not defined in *DLOAD; must be supplied before apply()."""
        assert gravity_load.density is None

    def test_gravity_vector_property(self, gravity_load):
        """gravity_vector should equal magnitude * direction."""
        gv = gravity_load.gravity_vector
        magnitude = self.EXPECTED["gravity_magnitude"]
        expected = tuple(magnitude * d for d in self.EXPECTED["gravity_direction"])
        for a, b in zip(gv, expected):
            assert a == pytest.approx(b, abs=1e-6)

    def test_gravity_load_name(self, gravity_load):
        assert gravity_load.name == "DLOAD_GRAV_EALL"


# ---------------------------------------------------------------------------
# Minimal .inp builder used by the unit-test classes below
# ---------------------------------------------------------------------------

_MINIMAL_INP = """\
*HEADING
Units: N, mm, t, s
*NODE
1, 0.0, 0.0, 0.0
2, 10.0, 0.0, 0.0
3, 0.0, 10.0, 0.0
4, 0.0, 0.0, 10.0
*ELEMENT, TYPE=C3D4, ELSET=PART
1, 1, 2, 3, 4
*NSET, NSET=FIX
1,
*NSET, NSET=LOAD_FACE
2, 3, 4
*BOUNDARY
FIX, 1, 3
*CLOAD
LOAD_FACE, 3, -50.0
"""

_INP_WITH_AMPLITUDE = """\
*HEADING
Test amplitude
*NODE
1, 0.0, 0.0, 0.0
2, 5.0, 0.0, 0.0
3, 0.0, 5.0, 0.0
4, 0.0, 0.0, 5.0
*ELEMENT, TYPE=C3D4, ELSET=ALL
1, 1, 2, 3, 4
*AMPLITUDE, NAME=RAMP
0.0, 0.0, 1.0, 1.0
*AMPLITUDE, NAME=STEP_AMP
0.0, 0.0, 0.001, 1.0, 1.0, 1.0
*NSET, NSET=FIX
1,
*BOUNDARY
FIX, 1, 3
"""

_INP_NO_UNIT_HINT = """\
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*ELEMENT, TYPE=C3D4, ELSET=ALL
1, 1, 2, 3, 4
*NSET, NSET=FIX
1,
*BOUNDARY
FIX, 1, 3
"""

_INP_UNKNOWN_KEYWORDS = """\
*HEADING
Test unknown keywords
*NODE
1, 0.0, 0.0, 0.0
2, 2.0, 0.0, 0.0
3, 0.0, 2.0, 0.0
4, 0.0, 0.0, 2.0
*ELEMENT, TYPE=C3D4, ELSET=ALL
1, 1, 2, 3, 4
*SOME_FUTURE_KEYWORD, PARAM=VALUE
data line 1
data line 2
*ANOTHER_UNSUPPORTED
data
*NSET, NSET=FIX
1,
*BOUNDARY
FIX, 1, 3
"""

_INP_WITH_CLOAD_DLOAD_AMPLITUDE = """\
*HEADING
Amplitude reference coverage
*NODE
1, 0.0, 0.0, 0.0
2, 10.0, 0.0, 0.0
3, 0.0, 10.0, 0.0
4, 0.0, 0.0, 10.0
*ELEMENT, TYPE=C3D4, ELSET=ALL
1, 1, 2, 3, 4
*NSET, NSET=LOADN
2,
*AMPLITUDE, NAME=RAMP
0.0, 0.0, 1.0, 1.0
*CLOAD, AMPLITUDE=RAMP
LOADN, 3, -50.0
*DLOAD, AMPLITUDE=RAMP
ALL, P3, -1.23
"""

_INP_WITH_MATERIAL_MAGNITUDE_HINT = """\
*HEADING
No explicit unit declaration
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*ELEMENT, TYPE=C3D4, ELSET=ALL
1, 1, 2, 3, 4
*MATERIAL, NAME=STEEL
*ELASTIC
210000.0, 0.3
*DENSITY
7.8e-09
*NSET, NSET=FIX
1,
*BOUNDARY
FIX, 1, 3
"""


class TestAmplitudeParsing:
    """*AMPLITUDE blocks are parsed and stored in meta['amplitudes']."""

    @pytest.fixture
    def load_case_with_amps(self, tmp_path):
        p = tmp_path / "amp.inp"
        p.write_text(_INP_WITH_AMPLITUDE)
        return AbaqusInpLoadCase.from_inp(str(p))

    def test_meta_has_amplitudes_key(self, load_case_with_amps):
        assert "amplitudes" in load_case_with_amps.meta

    def test_amplitude_names_present(self, load_case_with_amps):
        amps = load_case_with_amps.meta["amplitudes"]
        assert "RAMP" in amps
        assert "STEP_AMP" in amps

    def test_ramp_amplitude_has_two_points(self, load_case_with_amps):
        ramp = load_case_with_amps.meta["amplitudes"]["RAMP"]
        assert len(ramp) == 2

    def test_ramp_amplitude_values(self, load_case_with_amps):
        ramp = load_case_with_amps.meta["amplitudes"]["RAMP"]
        assert ramp[0] == pytest.approx((0.0, 0.0))
        assert ramp[1] == pytest.approx((1.0, 1.0))

    def test_step_amplitude_has_three_points(self, load_case_with_amps):
        step = load_case_with_amps.meta["amplitudes"]["STEP_AMP"]
        assert len(step) == 3

    def test_no_amplitudes_returns_empty_dict(self, tmp_path):
        p = tmp_path / "noamp.inp"
        p.write_text(_MINIMAL_INP)
        lc = AbaqusInpLoadCase.from_inp(str(p))
        assert lc.meta["amplitudes"] == {}


class TestUnitDetection:
    """Unit ambiguity is always warned; heading hint is detected when present."""

    @pytest.fixture
    def load_case_with_hint(self, tmp_path):
        p = tmp_path / "with_hint.inp"
        p.write_text(_MINIMAL_INP)  # heading says "N, mm, t, s"
        return AbaqusInpLoadCase.from_inp(str(p))

    @pytest.fixture
    def load_case_no_hint(self, tmp_path):
        p = tmp_path / "no_hint.inp"
        p.write_text(_INP_NO_UNIT_HINT)
        return AbaqusInpLoadCase.from_inp(str(p))

    def test_unit_warning_emitted_with_hint(self, tmp_path, caplog):
        import logging

        p = tmp_path / "hint.inp"
        p.write_text(_MINIMAL_INP)
        with caplog.at_level(
            logging.WARNING, logger="rapidcadpy.fea.load_case.abaqus_inp_load_case"
        ):
            AbaqusInpLoadCase.from_inp(str(p))
        assert any("unit" in r.message.lower() for r in caplog.records)

    def test_unit_warning_emitted_without_hint(self, tmp_path, caplog):
        import logging

        p = tmp_path / "nohint.inp"
        p.write_text(_INP_NO_UNIT_HINT)
        with caplog.at_level(
            logging.WARNING, logger="rapidcadpy.fea.load_case.abaqus_inp_load_case"
        ):
            AbaqusInpLoadCase.from_inp(str(p))
        assert any("unit" in r.message.lower() for r in caplog.records)

    def test_no_hint_message_mentions_verify(self, tmp_path, caplog):
        import logging

        p = tmp_path / "nohint2.inp"
        p.write_text(_INP_NO_UNIT_HINT)
        with caplog.at_level(
            logging.WARNING, logger="rapidcadpy.fea.load_case.abaqus_inp_load_case"
        ):
            AbaqusInpLoadCase.from_inp(str(p))
        msgs = " ".join(r.message for r in caplog.records)
        assert "verify" in msgs.lower() or "no unit hint" in msgs.lower()

    def test_heading_hint_detected_in_meta(self, load_case_with_hint):
        hint = load_case_with_hint.meta.get("unit_hint_from_heading")
        assert hint is not None
        assert "mm" in hint.lower() or "n-mm" in hint.lower() or "mpa" in hint.lower()

    def test_no_heading_hint_is_none_in_meta(self, load_case_no_hint):
        assert load_case_no_hint.meta.get("unit_hint_from_heading") is None

    def test_assumed_units_key_present(self, load_case_with_hint):
        assert "assumed_units" in load_case_with_hint.meta

    def test_assumed_units_mentions_mm(self, load_case_with_hint):
        assert "mm" in load_case_with_hint.meta["assumed_units"].lower()


class TestAmplitudeReferenceFlagging:
    """Amplitude references on load keywords are captured and explicitly flagged."""

    def test_cload_amplitude_reference_is_captured(self, tmp_path):
        p = tmp_path / "amp_refs.inp"
        p.write_text(_INP_WITH_CLOAD_DLOAD_AMPLITUDE)
        lc = AbaqusInpLoadCase.from_inp(str(p))

        refs = lc.meta.get("cload_amplitude_references", [])
        assert len(refs) == 1
        assert refs[0]["amplitude"] == "RAMP"
        assert refs[0]["target"] == "LOADN"

    def test_dload_entries_and_amplitude_refs_are_captured(self, tmp_path):
        p = tmp_path / "dload_refs.inp"
        p.write_text(_INP_WITH_CLOAD_DLOAD_AMPLITUDE)
        lc = AbaqusInpLoadCase.from_inp(str(p))

        dloads = lc.meta.get("dload_entries", [])
        assert len(dloads) == 1
        assert dloads[0]["target"] == "ALL"
        assert dloads[0]["label"] == "P3"

        dload_refs = lc.meta.get("dload_amplitude_references", [])
        assert len(dload_refs) == 1
        assert dload_refs[0]["amplitude"] == "RAMP"


class TestMaterialMagnitudeUnitHint:
    """Material property magnitudes contribute to unit detection metadata."""

    def test_material_hint_present_when_elastic_density_present(self, tmp_path):
        p = tmp_path / "mat_units.inp"
        p.write_text(_INP_WITH_MATERIAL_MAGNITUDE_HINT)
        lc = AbaqusInpLoadCase.from_inp(str(p))

        assert lc.meta.get("unit_hint_from_material_magnitude") is not None
        assert "detected_units" in lc.meta
        assert "unit_correction_options" in lc.meta
        assert len(lc.meta["unit_correction_options"]) >= 3

    def test_material_properties_are_stored_in_meta(self, tmp_path):
        p = tmp_path / "mat_props.inp"
        p.write_text(_INP_WITH_MATERIAL_MAGNITUDE_HINT)
        lc = AbaqusInpLoadCase.from_inp(str(p))

        props = lc.meta.get("material_properties", {})
        assert "elastic" in props
        assert "density" in props
        assert len(props["elastic"]) >= 1
        assert len(props["density"]) >= 1


class TestUnsupportedKeywordWarning:
    """Unknown keywords produce a named debug log entry, not a crash or silent skip."""

    def test_unknown_keyword_does_not_crash(self, tmp_path):
        p = tmp_path / "unknown_kw.inp"
        p.write_text(_INP_UNKNOWN_KEYWORDS)
        lc = AbaqusInpLoadCase.from_inp(str(p))
        assert isinstance(lc, LoadCase)

    def test_mesh_still_parsed_despite_unknown_keywords(self, tmp_path):
        p = tmp_path / "unknown_kw2.inp"
        p.write_text(_INP_UNKNOWN_KEYWORDS)
        lc = AbaqusInpLoadCase.from_inp(str(p))
        assert lc.mesh_nodes is not None
        assert lc.mesh_nodes.shape == (4, 3)

    def test_unknown_keyword_logged(self, tmp_path, caplog):
        import logging

        p = tmp_path / "unknown_kw3.inp"
        p.write_text(_INP_UNKNOWN_KEYWORDS)
        with caplog.at_level(
            logging.DEBUG, logger="rapidcadpy.fea.load_case.abaqus_inp_load_case"
        ):
            AbaqusInpLoadCase.from_inp(str(p))
        msgs = " ".join(r.message for r in caplog.records)
        # Both unknown keywords should appear by name
        assert "SOME_FUTURE_KEYWORD" in msgs
        assert "ANOTHER_UNSUPPORTED" in msgs

    def test_known_skip_keywords_do_not_appear_as_unknown(self, tmp_path, caplog):
        """MATERIAL, STEP etc. are in _SKIP_KEYWORDS and must NOT be logged as unknown."""
        import logging

        inp = """\
*HEADING
bench
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 0.0, 1.0, 0.0
4, 0.0, 0.0, 1.0
*ELEMENT, TYPE=C3D4, ELSET=ALL
1, 1, 2, 3, 4
*MATERIAL, NAME=STEEL
*ELASTIC
200000.0, 0.3
*SOLID SECTION, ELSET=ALL, MATERIAL=STEEL
*STEP
*STATIC
*END STEP
"""
        p = tmp_path / "known_skip.inp"
        p.write_text(inp)
        with caplog.at_level(
            logging.DEBUG, logger="rapidcadpy.fea.load_case.abaqus_inp_load_case"
        ):
            AbaqusInpLoadCase.from_inp(str(p))
        skip_msgs = [
            r.message for r in caplog.records if "Skipping unsupported" in r.message
        ]
        for msg in skip_msgs:
            assert "MATERIAL" not in msg
            assert "ELASTIC" not in msg
            assert "STEP" not in msg


class TestParseSummaryDetail:
    """The parse-summary log contains NSET names, DOF detail, load vectors."""

    @pytest.fixture
    def caplog_records(self, tmp_path, caplog):
        import logging

        p = tmp_path / "summary.inp"
        p.write_text(_MINIMAL_INP)
        with caplog.at_level(
            logging.INFO, logger="rapidcadpy.fea.load_case.abaqus_inp_load_case"
        ):
            AbaqusInpLoadCase.from_inp(str(p))
        return caplog.records

    def _summary_text(self, records):
        return " ".join(r.message for r in records if "Parsed Abaqus" in r.message)

    def test_summary_log_emitted(self, caplog_records):
        assert any("Parsed Abaqus" in r.message for r in caplog_records)

    def test_nset_names_in_summary(self, caplog_records):
        text = self._summary_text(caplog_records)
        assert "FIX" in text
        assert "LOAD_FACE" in text

    def test_bc_dofs_in_summary(self, caplog_records):
        text = self._summary_text(caplog_records)
        # BC for FIX covers DOFs 1-3
        assert "FIX" in text
        assert "1" in text and "2" in text and "3" in text

    def test_load_direction_in_summary(self, caplog_records):
        text = self._summary_text(caplog_records)
        # CLOAD is DOF 3, magnitude 50 N, direction -z
        assert "-z" in text or "50" in text

    def test_load_magnitude_in_summary(self, caplog_records):
        text = self._summary_text(caplog_records)
        assert "50" in text

    def test_amplitude_count_in_summary(self, tmp_path, caplog):
        import logging

        p = tmp_path / "amp_summary.inp"
        p.write_text(_INP_WITH_AMPLITUDE)
        with caplog.at_level(
            logging.INFO, logger="rapidcadpy.fea.load_case.abaqus_inp_load_case"
        ):
            AbaqusInpLoadCase.from_inp(str(p))
        text = " ".join(
            r.message for r in caplog.records if "Parsed Abaqus" in r.message
        )
        assert "RAMP" in text
        assert "STEP_AMP" in text
