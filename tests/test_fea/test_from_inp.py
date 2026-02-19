import pytest
import pathlib
import textwrap
import numpy as np

from rapidcadpy.fea.load_case import LoadCase
from rapidcadpy.fea.boundary_conditions import FixedConstraint, PointLoad


INP_FILE = pathlib.Path(__file__).parent.parent / "test_files" / "00004968.inp"


class TestFromInp:
    @pytest.fixture
    def get_inp_file(self):
        inp_file = pathlib.Path(__file__).parent.parent / "test_files"
        return inp_file / "00004968.inp"

    @pytest.fixture
    def load_case(self):
        """Parse the sample .inp file once and reuse across tests."""
        return LoadCase.from_inp(str(INP_FILE))

    # ------------------------------------------------------------------
    # Basic return type / structure
    # ------------------------------------------------------------------

    def test_returns_load_case_instance(self, load_case):
        assert isinstance(load_case, LoadCase)

    def test_problem_id_is_set(self, load_case):
        assert load_case.problem_id == "00004968"

    def test_description_contains_filename(self, load_case):
        assert "00004968.inp" in load_case.description

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

    def test_bounds_values_match_node_extents(self, load_case):
        """Bounds should span the node coordinate range in the file."""
        b = load_case.bounds
        # From the .inp file the extreme coords are ±0.235537, -0.661157..0, ±0.5
        assert b["x_min"] == pytest.approx(-0.235537, abs=1e-5)
        assert b["x_max"] == pytest.approx(0.235537, abs=1e-5)
        assert b["y_min"] == pytest.approx(-0.661157, abs=1e-5)
        assert b["y_max"] == pytest.approx(0.0, abs=1e-5)
        assert b["z_min"] == pytest.approx(-0.5, abs=1e-5)
        assert b["z_max"] == pytest.approx(0.5, abs=1e-5)

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def test_boundary_conditions_not_empty(self, load_case):
        assert len(load_case.boundary_conditions) > 0

    def test_boundary_condition_type_is_fixed_constraint(self, load_case):
        for bc in load_case.boundary_conditions:
            assert isinstance(bc, FixedConstraint)

    def test_fixed_constraint_locks_all_three_dofs(self, load_case):
        """CONSTRAINT set has BOUNDARY 1,3 → all three translational DOFs locked."""
        bc = load_case.boundary_conditions[0]
        assert bc.dofs == (True, True, True)

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def test_loads_not_empty(self, load_case):
        assert len(load_case.loads) > 0

    def test_load_type_is_point_load(self, load_case):
        for load in load_case.loads:
            assert isinstance(load, PointLoad)

    def test_load_force_is_in_z_direction(self, load_case):
        """CLOAD LOAD,3,-125 → force in Z, negative direction."""
        load = load_case.loads[0]
        fx, fy, fz = load.force
        assert fx == pytest.approx(0.0)
        assert fy == pytest.approx(0.0)
        assert fz == pytest.approx(-125.0, abs=1e-5)

    def test_load_magnitude_newtons(self, load_case):
        assert load_case.loads[0].magnitude_newtons == pytest.approx(125.0, abs=1e-5)

    # ------------------------------------------------------------------
    # Meta / selectors
    # ------------------------------------------------------------------

    def test_meta_is_populated(self, load_case):
        assert load_case.meta is not None

    def test_meta_contains_node_sets_count(self, load_case):
        # The file has ALL_NODES, CONSTRAINT, LOAD → at least 3
        assert load_case.meta["node_sets_count"] >= 2

    def test_selectors_populated(self, load_case):
        assert hasattr(load_case, "selectors")
        assert len(load_case.selectors) > 0

    def test_abaqus_element_type_detected(self, load_case):
        assert load_case.meta.get("abaqus_element_type") == "C3D4"

    # ------------------------------------------------------------------
    # Mesh data
    # ------------------------------------------------------------------

    def test_mesh_nodes_is_ndarray(self, load_case):
        assert isinstance(load_case.mesh_nodes, np.ndarray)

    def test_mesh_nodes_shape(self, load_case):
        # 24 nodes in the file, 3 coords each
        assert load_case.mesh_nodes.shape == (24, 3)

    def test_mesh_elements_is_ndarray(self, load_case):
        assert isinstance(load_case.mesh_elements, np.ndarray)

    def test_mesh_elements_shape(self, load_case):
        # 36 tet4 elements, 4 nodes each
        assert load_case.mesh_elements.shape == (36, 4)

    def test_mesh_element_type_is_tet4(self, load_case):
        assert load_case.mesh_element_type == "tet4"

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            LoadCase.from_inp("/nonexistent/path/to/file.inp")

    # ------------------------------------------------------------------
    # Minimal inline .inp fixture
    # ------------------------------------------------------------------

    @pytest.fixture
    def minimal_inp(self, tmp_path):
        """Write a tiny self-contained .inp file for isolated unit testing."""
        content = textwrap.dedent("""\
            *HEADING
            Minimal test model

            *NODE, NSET=ALL_NODES
            1, 0.0, 0.0, 0.0
            2, 1.0, 0.0, 0.0
            3, 0.0, 1.0, 0.0
            4, 0.0, 0.0, 1.0

            *ELEMENT, TYPE=C3D4, ELSET=ALL_ELEMENTS
            1, 1, 2, 3, 4

            *NSET, NSET=FIXED
            1

            *NSET, NSET=LOADED
            2

            *BOUNDARY
            FIXED, 1, 3

            *STEP
            *STATIC

            *CLOAD
            LOADED, 2, -500.0

            *END STEP
        """)
        inp_path = tmp_path / "minimal.inp"
        inp_path.write_text(content)
        return inp_path

    def test_minimal_problem_id(self, minimal_inp):
        lc = LoadCase.from_inp(str(minimal_inp))
        # from_inp uppercases the stem: Path.stem.upper()
        assert lc.problem_id == "MINIMAL"

    def test_minimal_bounds(self, minimal_inp):
        lc = LoadCase.from_inp(str(minimal_inp))
        b = lc.bounds
        assert b["x_min"] == pytest.approx(0.0)
        assert b["x_max"] == pytest.approx(1.0)
        assert b["y_min"] == pytest.approx(0.0)
        assert b["y_max"] == pytest.approx(1.0)
        assert b["z_min"] == pytest.approx(0.0)
        assert b["z_max"] == pytest.approx(1.0)

    def test_minimal_fixed_constraint(self, minimal_inp):
        lc = LoadCase.from_inp(str(minimal_inp))
        assert len(lc.boundary_conditions) == 1
        bc = lc.boundary_conditions[0]
        assert isinstance(bc, FixedConstraint)
        assert bc.dofs == (True, True, True)

    def test_minimal_point_load(self, minimal_inp):
        lc = LoadCase.from_inp(str(minimal_inp))
        assert len(lc.loads) == 1
        load = lc.loads[0]
        assert isinstance(load, PointLoad)
        _fx, fy, _fz = load.force
        assert fy == pytest.approx(-500.0)

    def test_minimal_mesh_nodes(self, minimal_inp):
        lc = LoadCase.from_inp(str(minimal_inp))
        assert isinstance(lc.mesh_nodes, np.ndarray)
        assert lc.mesh_nodes.shape == (4, 3)

    def test_minimal_mesh_elements(self, minimal_inp):
        lc = LoadCase.from_inp(str(minimal_inp))
        assert isinstance(lc.mesh_elements, np.ndarray)
        assert lc.mesh_elements.shape == (1, 4)

