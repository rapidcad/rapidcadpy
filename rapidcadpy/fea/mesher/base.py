"""
Abstract base class for FEA mesh generators.

This module defines the interface that all mesher implementations must follow,
enabling drop-in replacement of different meshing backends (Netgen, GMSH, etc.).
"""

from abc import ABC, abstractmethod
from typing import Literal, Tuple, Optional, Dict, Any
import torch
from pathlib import Path


class MesherBase(ABC):
    """
    Abstract base class for FEA mesh generators.
    
    All mesher implementations (Netgen, GMSH, etc.) should inherit from this class
    and implement the required methods. This enables easy swapping of meshing backends.
    """
    
    def __init__(self, num_threads: int = 0):
        """
        Initialize the mesher.
        
        Args:
            num_threads: Number of threads for parallel meshing (0 = auto-detect)
        """
        self.num_threads = num_threads
        if num_threads == 0:
            import multiprocessing
            self.num_threads = multiprocessing.cpu_count()
    
    @abstractmethod
    def generate_mesh(
        self,
        filename: str,
        mesh_size: float = 1.0,
        element_type: Literal["tet4", "tet10", "hex8", "hex20"] = "tet4",
        dim: int = 3,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mesh from geometry file.
        
        Args:
            filename: Path to geometry file (.step, .stp, .stl, etc.)
            mesh_size: Maximum element size
            element_type: Type of elements to generate
            dim: Spatial dimension (2 or 3)
            verbose: Print meshing progress
            **kwargs: Additional mesher-specific parameters
            
        Returns:
            Tuple of (nodes, elements):
                - nodes: Tensor of shape (n_nodes, dim) with nodal coordinates
                - elements: Tensor of shape (n_elements, nodes_per_element) with connectivity
                
        Raises:
            FileNotFoundError: If geometry file doesn't exist
            ValueError: If unsupported element type or file format
            RuntimeError: If meshing fails
        """
        pass
    
    @abstractmethod
    def get_supported_element_types(self) -> list[str]:
        """
        Get list of element types supported by this mesher.
        
        Returns:
            List of supported element type strings (e.g., ['tet4', 'tet10'])
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """
        Get list of geometry file formats supported by this mesher.
        
        Returns:
            List of supported file extensions (e.g., ['.step', '.stp', '.stl'])
        """
        pass
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Check if this mesher's dependencies are available.
        
        Returns:
            True if mesher can be used, False otherwise
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Get the name of this mesher.
        
        Returns:
            Human-readable mesher name (e.g., 'Netgen', 'GMSH')
        """
        pass
    
    def validate_inputs(
        self,
        filename: str,
        element_type: str,
        dim: int
    ) -> None:
        """
        Validate input parameters before meshing.
        
        Args:
            filename: Path to geometry file
            element_type: Element type to generate
            dim: Spatial dimension
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parameters are invalid
        """
        # Check file exists
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Geometry file not found: {filename}")
        
        # Check file format is supported
        suffix = filepath.suffix.lower()
        if suffix not in self.get_supported_formats():
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {', '.join(self.get_supported_formats())}"
            )
        
        # Check element type is supported
        if element_type not in self.get_supported_element_types():
            raise ValueError(
                f"Unsupported element type: {element_type}. "
                f"Supported types: {', '.join(self.get_supported_element_types())}"
            )
        
        # Check dimension
        if dim not in [2, 3]:
            raise ValueError(f"Dimension must be 2 or 3, got {dim}")
    
    def get_mesh_info(
        self,
        nodes: torch.Tensor,
        elements: torch.Tensor,
        element_type: str
    ) -> Dict[str, Any]:
        """
        Get information about the generated mesh.
        
        Args:
            nodes: Node coordinates tensor
            elements: Element connectivity tensor
            element_type: Element type
            
        Returns:
            Dictionary with mesh statistics
        """
        return {
            "n_nodes": nodes.shape[0],
            "n_elements": elements.shape[0],
            "element_type": element_type,
            "nodes_per_element": elements.shape[1],
            "dimension": nodes.shape[1],
            "bounding_box": {
                "min": nodes.min(dim=0).values.tolist(),
                "max": nodes.max(dim=0).values.tolist(),
            }
        }
