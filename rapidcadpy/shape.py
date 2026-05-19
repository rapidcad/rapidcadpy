from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

from .selectors import filter_linear_edges_by_selector

if TYPE_CHECKING:
    from .app import App
    from .fea.boundary_conditions import BoundaryCondition, Load
    from .fea.materials import MaterialProperties
    from .fea.results import FEAResults


class Shape(ABC):
    def __init__(self, obj, app: Optional["App"]) -> None:
        self.obj = obj
        self.app = app
        self.material: Optional[Union["MaterialProperties", str]] = "STEEL"
        self._selected_edges: Optional[List[Any]] = None
        self._selected_edge_selector: Optional[str] = None
        if app is not None:
            app.register_shape(self)

    @abstractmethod
    def volume(self) -> float:
        pass

    @abstractmethod
    def to_stl(self, file_name: str) -> None:
        pass

    @abstractmethod
    def to_step(self, file_name: str) -> None:
        pass

    @abstractmethod
    def to_png(
        self,
        file_name: str,
        view: str = "iso",
        width: int = 800,
        height: int = 600,
        backend: str = "auto",
    ) -> None:
        pass

    @abstractmethod
    def cut(self, other: "Shape") -> "Shape":
        pass

    @abstractmethod
    def union(self, other: "Shape") -> "Shape":
        pass

    # ------------------------------------------------------------------
    # Edge selection & fillet — backend-agnostic default implementation
    # ------------------------------------------------------------------

    @abstractmethod
    def _raw_edges(self) -> List[Any]:
        """Return all raw edge objects for this shape (backend-specific types)."""
        ...

    @abstractmethod
    def _is_linear_edge(self, edge: Any) -> bool:
        """Return True when *edge* is a straight line segment."""
        ...

    @abstractmethod
    def _edge_direction_vector(self, edge: Any) -> tuple:
        """Return the (dx, dy, dz) direction of a linear *edge*."""
        ...

    @abstractmethod
    def _apply_fillet_to_edges(self, edges: List[Any], radius: float) -> None:
        """Apply fillets of *radius* to *edges*, updating ``self.obj`` in-place."""
        ...

    def _clear_edge_selection(self) -> None:
        self._selected_edges = None
        self._selected_edge_selector = None

    def edges(self, selector: Optional[str] = None) -> "Shape":
        """Select edges by a string selector (e.g. ``"|Z"``) and return *self*.

        The selector logic is evaluated here in the base class, so all CAD
        kernels share identical filtering semantics.  Call ``.fillet()``
        immediately after to apply the rounded blend.
        """
        self._selected_edges = filter_linear_edges_by_selector(
            self._raw_edges(),
            selector,
            self._is_linear_edge,
            self._edge_direction_vector,
        )
        self._selected_edge_selector = selector
        return self

    def fillet(self, radius: float) -> "Shape":
        """Blend the previously selected (or all) edges with *radius*."""
        edges_to_fillet = (
            self._selected_edges
            if self._selected_edges is not None
            else self._raw_edges()
        )
        selector = self._selected_edge_selector
        selector_desc = selector or "all edges"
        # Clear before the operation so a failed call doesn't leave stale state.
        self._selected_edges = None
        self._selected_edge_selector = None

        if not edges_to_fillet:
            raise ValueError(f"No edges matched selector {selector_desc!r} for fillet")

        self._apply_fillet_to_edges(edges_to_fillet, radius, selector)
        return self

    def analyze(
        self,
        material: Union["MaterialProperties", str, None] = None,
        loads: Optional[List["Load"]] = None,
        constraints: Optional[List["BoundaryCondition"]] = None,
        mesh_size: float = 2.0,
        element_type: str = "tet4",
    ) -> "FEAResults":
        """
        Perform Finite Element Analysis on this shape.

        Args:
            material: Material properties or material name (defaults to shape.material or STEEL)
            loads: List of loads to apply
            constraints: List of boundary conditions
            mesh_size: Target mesh element size in mm
            element_type: Element type (solver-dependent, default: 'tet4')
            verbose: Print detailed analysis progress

        Returns:
            FEAResults object with stress, displacement, and analysis data

        Example:
            >>> from .fea import Material, DistributedLoad, FixedConstraint
            >>> beam = wp.rect(10, 10).extrude(100)
            >>> result = beam.analyze(
            ...     material=Material.ALUMINUM_6061_T6,
            ...     loads=[DistributedLoad('top', force=-1000)],
            ...     constraints=[FixedConstraint('end_1')]
            ... )
            >>> print(result.summary())
            >>> result.show()
        """
        # Handle material selection
        from .fea.fea_analyzer import FEAAnalyzer
        from .fea.materials import Material, MaterialProperties

        resolved_material: MaterialProperties
        if material is None:
            mat_or_str = self.material or Material.STEEL
            if isinstance(mat_or_str, str):
                resolved_material = getattr(
                    Material, mat_or_str.upper(), Material.STEEL
                )
            else:
                resolved_material = mat_or_str
        elif isinstance(material, str):
            # Look up material by name
            resolved_material = getattr(Material, material.upper(), Material.STEEL)
        else:
            resolved_material = material

        # Get analyzer from concrete implementation
        analyzer = FEAAnalyzer(
            shape=self,
            material=resolved_material,
            kernel="torch-fem",
            mesh_size=mesh_size,
        )

        if analyzer is None:
            raise NotImplementedError(
                f"FEA analysis not available for {self.__class__.__name__}. "
                f"Make sure FEA dependencies are installed: pip install rapidcadpy[fea]"
            )

        # Add loads and constraints
        for load in loads or []:
            analyzer.add_load(load)

        for constraint in constraints or []:
            analyzer.add_constraint(constraint)

        # Solve
        return analyzer.solve()

    @abstractmethod
    def volume(self) -> float: ...
