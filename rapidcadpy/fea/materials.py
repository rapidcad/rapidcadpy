"""
Material properties for FEA analysis.

This module provides pre-defined engineering materials and a custom material class.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MaterialProperties:
    """Material properties for FEA analysis"""

    name: str
    E: float  # Young's modulus (MPa)
    nu: float  # Poisson's ratio
    density: float  # Density (g/cm³)
    yield_strength: Optional[float] = None  # MPa
    ultimate_strength: Optional[float] = None  # MPa

    def to_torchfem(self):
        """Convert to torch-fem material format"""
        try:
            from torchfem.materials import IsotropicElasticity3D

            return IsotropicElasticity3D(E=self.E, nu=self.nu)
        except ImportError:
            raise ImportError(
                "torch-fem not available. Install with: pip install rapidcadpy[fea]"
            )


class Material:
    """Pre-defined engineering materials library"""

    # Steels
    STEEL = MaterialProperties(
        name="Structural Steel",
        E=200000.0,
        nu=0.30,
        density=7.85,
        yield_strength=250,
        ultimate_strength=400,
    )

    STAINLESS_304 = MaterialProperties(
        name="Stainless Steel 304",
        E=193000.0,
        nu=0.29,
        density=8.00,
        yield_strength=215,
        ultimate_strength=505,
    )

    STAINLESS_316 = MaterialProperties(
        name="Stainless Steel 316",
        E=193000.0,
        nu=0.29,
        density=8.00,
        yield_strength=290,
        ultimate_strength=580,
    )

    # Aluminum alloys
    ALUMINUM_6061_T6 = MaterialProperties(
        name="Aluminum 6061-T6",
        E=68900.0,
        nu=0.33,
        density=2.70,
        yield_strength=276,
        ultimate_strength=310,
    )

    ALUMINUM_7075_T6 = MaterialProperties(
        name="Aluminum 7075-T6",
        E=71700.0,
        nu=0.33,
        density=2.81,
        yield_strength=503,
        ultimate_strength=572,
    )

    ALUMINUM_2024_T3 = MaterialProperties(
        name="Aluminum 2024-T3",
        E=73100.0,
        nu=0.33,
        density=2.78,
        yield_strength=345,
        ultimate_strength=483,
    )

    # Titanium
    TITANIUM_6AL4V = MaterialProperties(
        name="Titanium Ti-6Al-4V",
        E=113800.0,
        nu=0.33,
        density=4.43,
        yield_strength=880,
        ultimate_strength=950,
    )

    TITANIUM_GRADE_2 = MaterialProperties(
        name="Titanium Grade 2",
        E=102700.0,
        nu=0.34,
        density=4.51,
        yield_strength=275,
        ultimate_strength=345,
    )

    # Other metals
    BRASS = MaterialProperties(
        name="Brass",
        E=100000.0,
        nu=0.34,
        density=8.50,
        yield_strength=200,
        ultimate_strength=300,
    )

    COPPER = MaterialProperties(
        name="Copper",
        E=117000.0,
        nu=0.33,
        density=8.96,
        yield_strength=70,
        ultimate_strength=220,
    )

    BRONZE = MaterialProperties(
        name="Bronze",
        E=103000.0,
        nu=0.34,
        density=8.80,
        yield_strength=150,
        ultimate_strength=380,
    )

    # Cast irons
    CAST_IRON = MaterialProperties(
        name="Gray Cast Iron",
        E=110000.0,
        nu=0.28,
        density=7.20,
        yield_strength=250,
        ultimate_strength=250,
    )

    @classmethod
    def list_materials(cls):
        """List all available pre-defined materials"""
        materials = []
        for attr_name in dir(cls):
            if not attr_name.startswith("_") and attr_name.isupper():
                attr = getattr(cls, attr_name)
                if isinstance(attr, MaterialProperties):
                    materials.append((attr_name, attr.name))
        return materials


class CustomMaterial(MaterialProperties):
    """
    User-defined custom material.

    Example:
        >>> titanium = CustomMaterial(
        ...     name="Custom Ti Alloy",
        ...     E=110000.0,
        ...     nu=0.33,
        ...     density=4.43,
        ...     yield_strength=900
        ... )
    """

    pass
