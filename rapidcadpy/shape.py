from abc import ABC, abstractmethod
from ast import Load
from typing import TYPE_CHECKING, List, Optional, Union

from rapidcadpy.fea.boundary_conditions import BoundaryCondition
from rapidcadpy.fea.kernels.base import FEAAnalyzer
from rapidcadpy.fea.materials import Material, MaterialProperties
from rapidcadpy.fea.results import FEAResults

if TYPE_CHECKING:
    from rapidcadpy.app import App


class Shape(ABC):
    def __init__(self, obj, app: Optional["App"]) -> None:
        self.obj = obj
        self.app = app
        self.material: Optional[Union[MaterialProperties, str]] = Material.STEEL
        if app is not None:
            app.register_shape(self)

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
            >>> from rapidcadpy.fea import Material, DistributedLoad, FixedConstraint
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
            shape=self, material=resolved_material, kernel="torch-fem"
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
