"""Minimal in-process GMSH mesher."""

from pathlib import Path
from typing import Literal, Tuple

import torch

from .base import MesherBase


class GmshMesher(MesherBase):
    @classmethod
    def is_available(cls) -> bool:
        try:
            import gmsh  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def get_name(cls) -> str:
        return "GMSH (python)"

    def get_supported_element_types(self) -> list[str]:
        return ["tet4", "tet10"]

    def get_supported_formats(self) -> list[str]:
        return [".step", ".stp", ".brep", ".iges", ".igs"]

    def generate_mesh(
        self,
        filename: str,
        mesh_size: float = 1.0,
        element_type: Literal["tet4", "tet10", "hex8", "hex20"] = "tet4",
        dim: int = 3,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.validate_inputs(filename, element_type, dim)
        if dim != 3:
            raise ValueError("GmshMesher currently only supports 3D meshing")

        import gmsh

        initialized_here = False
        try:
            if not gmsh.isInitialized():
                gmsh.initialize()
                initialized_here = True

            if not verbose:
                gmsh.option.setNumber("General.Terminal", 0)

            gmsh.model.add("mesh")
            gmsh.model.occ.importShapes(str(Path(filename).resolve()))
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)

            if element_type == "tet10":
                gmsh.model.mesh.setOrder(2)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes = node_coords.reshape(-1, 3)

            elem_type_code = 4 if element_type == "tet4" else 11
            _, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type_code)

            if elem_node_tags.size == 0:
                raise RuntimeError(f"No {element_type} elements generated")

            nodes_per_elem = 4 if element_type == "tet4" else 10
            elements = elem_node_tags.reshape(-1, nodes_per_elem)

            node_tag_map = {int(tag): i for i, tag in enumerate(node_tags)}
            elements_remapped = [
                [node_tag_map[int(tag)] for tag in elem] for elem in elements
            ]

            return (
                torch.tensor(nodes, dtype=torch.float32),
                torch.tensor(elements_remapped, dtype=torch.int64),
            )
        except Exception as e:
            raise RuntimeError(f"In-process GMSH meshing failed: {e}")
        finally:
            if initialized_here:
                try:
                    gmsh.finalize()
                except Exception:
                    pass
