from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

from .selectors import filter_linear_edges_by_selector

if TYPE_CHECKING:
    from .app import App
    from .cad_objects import CadDocument, CadFeature
    from .fea.boundary_conditions import BoundaryCondition, Load
    from .fea.materials import MaterialProperties
    from .fea.results import FEAResults


class Shape(ABC):
    def __init__(
        self,
        obj,
        app: Optional["App"],
        document: Optional["CadDocument"] = None,
        feature: Optional["CadFeature"] = None,
    ) -> None:
        if feature is not None and document is not feature.document:
            raise ValueError("Shape document must own the bound feature.")
        self.obj = obj
        self.app = app
        self._document = document
        self._feature = feature
        self.material: Optional[Union["MaterialProperties", str]] = "STEEL"
        self._selected_edges: Optional[List[Any]] = None
        self._selected_edge_selector: Optional[str] = None
        if app is not None:
            app.register_shape(self)

    @property
    def document(self) -> Optional["CadDocument"]:
        """Backend-neutral reference to the owning native document."""
        return self._document

    @property
    def feature(self) -> Optional["CadFeature"]:
        """Backend-neutral reference to the native feature producing this shape."""
        return self._feature

    @property
    def is_parametric(self) -> bool:
        """Whether this shape is still connected to a live native feature."""
        return (
            self._document is not None
            and self._feature is not None
            and self._feature.is_parametric
        )

    def bind_native(
        self,
        document: Optional["CadDocument"],
        feature: Optional["CadFeature"],
    ) -> None:
        """Bind this shape to a native document and feature."""
        if feature is not None and document is not feature.document:
            raise ValueError("Shape document must own the bound feature.")
        self._document = document
        self._feature = feature

    def refresh_from_feature(self) -> bool:
        """Refresh ``obj`` from the bound native feature's current result.

        Returns ``True`` when a live feature shape was available.
        """
        if self._feature is None:
            return False
        native_shape = self._feature.shape
        if native_shape is None:
            return False
        self.obj = native_shape
        return True

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

    def intersection_volume(self, other: "Shape") -> float:
        """Return the volume shared by this solid and ``other``.

        Unlike :meth:`cut` and :meth:`union`, this is an inspection operation:
        it must not alter either operand or its feature history.  CAD backends
        with an exact solid-boolean implementation override it.  Keeping the
        method here makes assembly validation depend on the public RapidCADPy
        API rather than on a particular OCC/FreeCAD native object.
        """
        raise NotImplementedError(
            f"Exact intersection-volume queries are not implemented for "
            f"{self.__class__.__name__}."
        )

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
    def _apply_fillet_to_edges(
        self,
        edges: List[Any],
        radius: float,
        selector: Optional[str] = None,
    ) -> None:
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

    def export(self, file_name: str) -> None:
        """Export using the format implied by *file_name*.

        This compatibility entry point is documented by RapidCADPy and keeps
        callers independent of the backend-specific ``to_*`` methods.
        """

        suffix = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
        if suffix in {"step", "stp"}:
            self.to_step(file_name)
        elif suffix == "stl":
            self.to_stl(file_name)
        elif suffix == "png":
            self.to_png(file_name)
        else:
            raise ValueError(
                f"Unsupported export format for {file_name!r}; "
                "expected .step, .stp, .stl, or .png"
            )

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
