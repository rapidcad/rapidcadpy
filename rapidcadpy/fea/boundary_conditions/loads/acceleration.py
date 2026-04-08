"""
AccelerationLoad — body-force load from gravity, centrifugal, or inertial acceleration.
"""

from typing import Optional, Tuple, Literal

from .base import Load


class AccelerationLoad(Load):
    """
    Body-force load due to acceleration (gravity, centrifugal, or inertial loading).

    This class converts a field acceleration (mm/s² or rad²/s²) into equivalent
    nodal forces by distributing element masses to their corner nodes (lumped-mass
    approach).  It covers the three forms that appear in Abaqus / CalculiX
    ``*DLOAD`` sections:

    **Gravity** — constant acceleration vector applied to the whole body:

    .. code-block:: text

        *DLOAD
        EALL, GRAV, 9810., 0., 0., -1.

    Nodal force: ``F_i = m_i × magnitude × direction``.

    **Centrifugal** — rotational body force for spinning components:

    .. code-block:: text

        *DLOAD
        EALL, CENTRIF, ω², px, py, pz, ax, ay, az

    Nodal force: ``F_i = m_i × ω² × r_i``  where ``r_i`` is the outward
    radial vector from the rotation axis to node *i*.

    **Body force** — general distributed force per unit volume (N/mm³):

    .. code-block:: text

        *DLOAD
        EALL, BX, 100.

    Nodal force: ``F_i = V_i × magnitude × direction`` (no density required).

    Unit conventions (mm / N / MPa system):

    - ``density`` in **t/mm³**.  Convert from g/cm³ by multiplying by ``1e-9``
      (e.g. steel 7.85 g/cm³ → ``7.85e-9`` t/mm³).
    - ``magnitude`` for gravity: mm/s² (Earth gravity = **9810** mm/s²).
    - ``magnitude`` for centrifugal: rad²/s² = ``(2π × RPM / 60)²``.
    - ``magnitude`` for body_force: N/mm³.

    Physical interpretation:

    *Gravity* creates bending moments proportional to the mass distribution.
    Sections far from the fixed support contribute disproportionately: removing
    material at the distal (free) end simultaneously reduces the inertial load
    and its moment arm — doubly beneficial under self-weight loading.

    *Centrifugal* stress grows with radius²: a node at twice the radius carries
    four times the centrifugal load.  Topology optimisation under centrifugal
    loads tends to remove rim material and retain a stiffer hub.
    """

    GRAVITY = "gravity"
    CENTRIFUGAL = "centrifugal"
    BODY_FORCE = "body_force"

    def __init__(
        self,
        load_type: Literal["gravity", "centrifugal", "body_force"],
        magnitude: float,
        density: Optional[float] = None,
        direction: Optional[Tuple[float, float, float]] = None,
        axis_point: Optional[Tuple[float, float, float]] = None,
        axis_direction: Optional[Tuple[float, float, float]] = None,
        element_set: str = "EALL",
    ):
        """
        Args:
            load_type: One of ``'gravity'``, ``'centrifugal'``, or ``'body_force'``.
            magnitude:
                - ``gravity``:     acceleration (mm/s²). Earth = 9810.
                - ``centrifugal``: ω² (rad²/s²) = ``(2π × RPM / 60)²``.
                - ``body_force``:  force per unit volume (N/mm³).
            density:
                Material density in **t/mm³** — required for ``gravity`` and
                ``centrifugal``.  You may pass a
                :class:`~rapidcadpy.fea.materials.MaterialProperties` instance
                directly; its ``density`` (g/cm³) is auto-converted.
            direction:
                Unit acceleration vector ``(dx, dy, dz)`` — required for
                ``gravity`` and ``body_force``.  Example downward gravity:
                ``(0., 0., -1.)``.
            axis_point:
                Point on the rotation axis (mm) — required for ``centrifugal``.
            axis_direction:
                Unit vector along the rotation axis — required for ``centrifugal``.
            element_set:
                Informational element-set name (reserved for future filtering).
        """
        load_type = load_type.lower()
        if load_type not in (self.GRAVITY, self.CENTRIFUGAL, self.BODY_FORCE):
            raise ValueError(
                f"load_type must be 'gravity', 'centrifugal', or 'body_force', "
                f"got {load_type!r}"
            )
        self.load_type = load_type
        self.magnitude = magnitude
        self.element_set = element_set

        # Accept a MaterialProperties object; convert g/cm³ → t/mm³
        if density is not None and hasattr(density, "density"):
            density = density.density * 1e-9
        self.density = density  # t/mm³, or None for body_force

        if direction is not None:
            direction = tuple(float(v) for v in direction)
        self.direction = direction

        if axis_point is not None:
            axis_point = tuple(float(v) for v in axis_point)
        self.axis_point = axis_point

        if axis_direction is not None:
            axis_direction = tuple(float(v) for v in axis_direction)
        self.axis_direction = axis_direction

        # Validate per load type
        # NOTE: density is required at apply() time, not at construction, so that
        # AccelerationLoad objects can be created from parsed INP files before
        # material properties are assigned.
        if self.load_type in (self.GRAVITY, self.BODY_FORCE):
            if self.direction is None:
                raise ValueError(
                    f"'direction' is required for load_type={self.load_type!r}."
                )
        if self.load_type == self.CENTRIFUGAL:
            if self.axis_point is None or self.axis_direction is None:
                raise ValueError(
                    "'axis_point' and 'axis_direction' are required for "
                    "load_type='centrifugal'."
                )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self.load_type == self.CENTRIFUGAL:
            import math as _math

            rpm = (_math.sqrt(self.magnitude) * 60.0) / (2.0 * _math.pi)
            extra = (
                f", axis_point={self.axis_point!r}, "
                f"axis_direction={self.axis_direction!r}, "
                f"≈{rpm:.0f} RPM"
            )
        else:
            extra = f", direction={self.direction!r}"
        return (
            f"AccelerationLoad(load_type={self.load_type!r}, "
            f"magnitude={self.magnitude!r}{extra}, "
            f"element_set={self.element_set!r})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gravity_vector(self) -> Tuple[float, float, float]:
        """
        Signed acceleration vector ``magnitude × direction`` (mm/s²).

        This is the gravitational acceleration experienced by every mass element,
        expressed as a Cartesian vector.
        """
        if self.direction is None:
            raise ValueError("direction is required for gravity/body_force types.")
        return tuple(self.magnitude * d for d in self.direction)

    @property
    def rpm(self) -> float:
        """
        Equivalent rotational speed in RPM (centrifugal loads only).

        Converts ω² stored in ``magnitude`` via
        ``RPM = sqrt(ω²) × 60 / (2π)``.
        Engineers reason in RPM; this bridges the Abaqus CENTRIF format and
        human-readable reporting.
        """
        if self.load_type != self.CENTRIFUGAL:
            raise AttributeError("rpm is only defined for centrifugal loads.")
        import math as _math

        return (_math.sqrt(self.magnitude) * 60.0) / (2.0 * _math.pi)

    # ------------------------------------------------------------------
    # Volume / mass helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tet4_volumes(nodes, elements):
        """
        Volumes of linear 4-node tetrahedral elements via scalar triple product.

        ``V = |det([v1, v2, v3])| / 6``  where v1, v2, v3 are edge vectors
        from node 0.

        Returns:
            Tensor (n_elements,) — volumes in mm³.
        """
        import torch

        v0 = nodes[elements[:, 0]]
        v1 = nodes[elements[:, 1]] - v0
        v2 = nodes[elements[:, 2]] - v0
        v3 = nodes[elements[:, 3]] - v0
        cross = torch.linalg.cross(v2, v3)
        return torch.abs((v1 * cross).sum(dim=1)) / 6.0

    @staticmethod
    def _hex8_volumes(nodes, elements):
        """
        Approximate volumes of 8-node hexahedral elements via 5-tet decomposition.

        The result is exact for regular brick meshes; for distorted hexes it is
        a first-order approximation sufficient for body-force lumping.

        Returns:
            Tensor (n_elements,) — volumes in mm³.
        """
        import torch

        tet_splits = [
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [3, 4, 6, 7],
            [1, 4, 5, 6],
            [1, 3, 4, 6],
        ]
        total = torch.zeros(len(elements), dtype=nodes.dtype, device=nodes.device)
        for s in tet_splits:
            v0 = nodes[elements[:, s[0]]]
            v1 = nodes[elements[:, s[1]]] - v0
            v2 = nodes[elements[:, s[2]]] - v0
            v3 = nodes[elements[:, s[3]]] - v0
            cross = torch.linalg.cross(v2, v3)
            total += torch.abs((v1 * cross).sum(dim=1)) / 6.0
        return total

    def _element_volumes(self, nodes, elements):
        """
        Dispatch to the correct volume calculator for the element topology.

        Supported: 4-node tet, 8-node hex, 10-node quad tet (corners only),
        20-node quad hex (corners only).
        """
        npe = elements.shape[1]
        if npe == 4:
            return self._tet4_volumes(nodes, elements)
        elif npe == 8:
            return self._hex8_volumes(nodes, elements)
        elif npe == 10:
            return self._tet4_volumes(nodes, elements[:, :4])
        elif npe == 20:
            return self._hex8_volumes(nodes, elements[:, :8])
        else:
            raise ValueError(
                f"AccelerationLoad does not support {npe}-node elements. "
                "Supported: 4-node tet, 8-node hex, 10-node quad tet, 20-node quad hex."
            )

    def _lumped_nodal_masses(self, nodes, elements):
        """
        Lumped nodal masses (t) via equal distribution to corner nodes.

        ``m_element = ρ · V``; each of the *npe* corner nodes receives
        ``m_element / npe``.  This matches the Abaqus default lumped-mass
        matrix for solid elements and is appropriate for static body-force
        computation.

        Returns:
            Tensor (n_nodes,) — nodal masses in tonnes.
        """
        import torch

        n_nodes = nodes.shape[0]
        npe = elements.shape[1]
        volumes = self._element_volumes(nodes, elements)
        elem_masses = self.density * volumes
        nodal_masses = torch.zeros(n_nodes, dtype=nodes.dtype, device=nodes.device)
        share = elem_masses / float(npe)
        for j in range(npe):
            nodal_masses.scatter_add_(0, elements[:, j], share)
        return nodal_masses

    def _lumped_nodal_volumes(self, nodes, elements):
        """
        Lumped nodal tributary volumes (mm³) for body-force loads.

        No density is involved; each node receives an equal share of each
        adjacent element's volume.

        Returns:
            Tensor (n_nodes,) — tributary volumes in mm³.
        """
        import torch

        n_nodes = nodes.shape[0]
        npe = elements.shape[1]
        volumes = self._element_volumes(nodes, elements)
        nodal_vols = torch.zeros(n_nodes, dtype=nodes.dtype, device=nodes.device)
        share = volumes / float(npe)
        for j in range(npe):
            nodal_vols.scatter_add_(0, elements[:, j], share)
        return nodal_vols

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, model, nodes, elements, geometry_info, mesh_size: float):
        """
        Distribute the acceleration body force into ``model.forces``.

        Physical laws applied:

        - **Gravity**:     ``F_i = m_i · a``  where ``a = magnitude × direction``.
        - **Body force**:  ``F_i = V_i · q``  where ``q = magnitude × direction (N/mm³)``.
        - **Centrifugal**: ``F_i = m_i · ω² · r_i``  where ``r_i`` is the
          perpendicular radial vector from the rotation axis to node *i*.
          Produces outward radial forces that penalise material at large radii.

        Args:
            model:         FEA solver model; must expose a ``forces`` tensor
                           of shape (n_nodes, 3).
            nodes:         Coordinate tensor (n_nodes, 3) in mm.
            elements:      Connectivity tensor (n_elements, nodes_per_element).
            geometry_info: Dict with ``bounding_box`` (unused here; kept for
                           API consistency).
            mesh_size:     Characteristic element size in mm (unused here; kept
                           for API consistency).

        Returns:
            int: Total number of nodes (all nodes receive a force contribution).
        """
        import torch

        n_nodes = nodes.shape[0]

        if self.load_type in (self.GRAVITY, self.CENTRIFUGAL) and self.density is None:
            raise ValueError(
                f"'density' (t/mm³) is required to apply load_type={self.load_type!r}. "
                "Pass a MaterialProperties instance or a numeric density in t/mm³."
            )

        if self.load_type == self.GRAVITY:
            nodal_masses = self._lumped_nodal_masses(nodes, elements)
            accel = torch.tensor(
                [self.magnitude * d for d in self.direction],
                dtype=nodes.dtype,
                device=nodes.device,
            )
            forces = nodal_masses.unsqueeze(1) * accel.unsqueeze(0)
            model.forces += forces

        elif self.load_type == self.BODY_FORCE:
            nodal_vols = self._lumped_nodal_volumes(nodes, elements)
            force_density = torch.tensor(
                [self.magnitude * d for d in self.direction],
                dtype=nodes.dtype,
                device=nodes.device,
            )
            forces = nodal_vols.unsqueeze(1) * force_density.unsqueeze(0)
            model.forces += forces

        elif self.load_type == self.CENTRIFUGAL:
            nodal_masses = self._lumped_nodal_masses(nodes, elements)
            ax = torch.tensor(
                self.axis_direction, dtype=nodes.dtype, device=nodes.device
            )
            ax_norm = ax / (torch.norm(ax) + 1e-15)
            pt = torch.tensor(self.axis_point, dtype=nodes.dtype, device=nodes.device)
            delta = nodes - pt.unsqueeze(0)
            proj = (delta * ax_norm.unsqueeze(0)).sum(dim=1, keepdim=True)
            radial = delta - proj * ax_norm.unsqueeze(0)
            forces = self.magnitude * nodal_masses.unsqueeze(1) * radial
            model.forces += forces

        return n_nodes
