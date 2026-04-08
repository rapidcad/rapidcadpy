"""
Abstract base class for all FEA load types.
"""

from abc import ABC, abstractmethod


class Load(ABC):
    """
    Abstract base class for all nodal and body loads.

    Each concrete subclass must implement :meth:`apply`, which distributes the
    physical load into ``model.forces`` using the mesh geometry.
    """

    @abstractmethod
    def apply(self, model, nodes, elements, geometry_info):
        """
        Apply this load to the FEA model.

        Args:
            model: FEA solver model object with a ``forces`` tensor.
            nodes: Mesh nodes tensor (n_nodes, 3).
            elements: Mesh elements tensor (n_elements, nodes_per_element).
            geometry_info: Dict with ``bounding_box`` and geometry information.

        Returns:
            int: Number of nodes to which forces were applied.
        """
        pass
