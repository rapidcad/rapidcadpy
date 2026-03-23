import pytest
import pathlib
import textwrap
import numpy as np

from ...rapidcadpy.fea.load_case.load_case import LoadCase
from ...rapidcadpy.fea.boundary_conditions import FixedConstraint, PointLoad


class TestFromInp:
    @pytest.fixture(params=["fea_test_4", "fea_test_5"])
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
            "fea_test_5": {
                "path": inp_file / "fea_test_5_freecad.inp",
                "problem_id": "FEA_TEST_5_FREECAD",
                "description_contains": "fea_test_5_freecad.inp",
                "bounds": {
                    "x_min": 0.0,
                    "x_max": 8000.0,
                    "y_min": 0.0,
                    "y_max": 1000.0,
                    "z_min": 0.0,
                    "z_max": 1000.0,
                },
                "bc_dofs": (True, True, True),
                "load_force": (0.0, 0.0, -9000000.0),
                "load_magnitude": 9000000.0,
                "node_sets_count_min": 1,
                "abaqus_element_type": None,
                "mesh_nodes_shape": (569, 3),
                "mesh_elements_shape": (242, 10),
                "mesh_element_type": "tet10",
            },
        }

        return cases[request.param]
    
    @pytest.fixture
    def load_case(self, get_inp_file) -> LoadCase:
        """Parse the sample .inp file once and reuse across tests."""
        return LoadCase.from_inp(str(get_inp_file["path"]))

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
        for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
            assert b[key] == pytest.approx(expected[key], abs=1e-5)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def test_boundary_conditions_not_empty(self, load_case):
        assert len(load_case.boundary_conditions) > 0

    def test_boundary_condition_type_is_fixed_constraint(self, load_case):
        for bc in load_case.boundary_conditions:
            assert isinstance(bc, FixedConstraint)

    def test_fixed_constraint_locks_all_three_dofs(self, load_case, get_inp_file):
        bc = load_case.boundary_conditions[0]
        assert bc.dofs == get_inp_file["bc_dofs"]

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def test_loads_not_empty(self, load_case):
        assert len(load_case.loads) > 0

    def test_load_type_is_point_load(self, load_case):
        for load in load_case.loads:
            assert isinstance(load, PointLoad)

    def test_load_force_is_in_z_direction(self, load_case, get_inp_file):
        load = load_case.loads[0]
        expected_fx, expected_fy, expected_fz = get_inp_file["load_force"]
        fx, fy, fz = load.force
        assert fx == pytest.approx(expected_fx, abs=1e-5)
        assert fy == pytest.approx(expected_fy, abs=1e-5)
        assert fz == pytest.approx(expected_fz, abs=1e-5)

    def test_load_magnitude_newtons(self, load_case, get_inp_file):
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

    def test_selectors_populated(self, load_case):
        assert hasattr(load_case, "selectors")
        assert len(load_case.selectors) > 0

    def test_abaqus_element_type_detected(self, load_case, get_inp_file):
        assert load_case.meta.get("abaqus_element_type") == get_inp_file["abaqus_element_type"]

    # ------------------------------------------------------------------
    # Mesh data
    # ------------------------------------------------------------------

    def test_mesh_nodes_is_ndarray(self, load_case):
        assert isinstance(load_case.mesh_nodes, np.ndarray)

    def test_mesh_nodes_shape(self, load_case, get_inp_file):
        assert load_case.mesh_nodes.shape == get_inp_file["mesh_nodes_shape"]

    def test_mesh_elements_is_ndarray(self, load_case):
        assert isinstance(load_case.mesh_elements, np.ndarray)

    def test_mesh_elements_shape(self, load_case, get_inp_file):
        assert load_case.mesh_elements.shape == get_inp_file["mesh_elements_shape"]

    def test_mesh_element_type_is_tet4(self, load_case, get_inp_file):
        assert load_case.mesh_element_type == get_inp_file["mesh_element_type"]

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            LoadCase.from_inp("/nonexistent/path/to/file.inp")

    # ---
    # Visualization
    ###
    def test_plot_fea(self, load_case):
        fea = load_case.get_fea_analyzer()
        fea.show("conditions", filename="test_plot_fea_conditions.png")



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
