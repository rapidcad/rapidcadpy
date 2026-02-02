"""
Example: Using Different Meshers with TorchFEMKernel

This example demonstrates how to use the abstract mesher interface to
swap between different meshing backends (Netgen, GMSH, etc.) as drop-in
replacements in the FEA workflow.
"""

from rapidcadpy import Sketch
from rapidcadpy.fea.mesher import NetgenMesher, MesherBase
from rapidcadpy.fea.kernels.torch_fem_kernel import TorchFEMKernel
from rapidcadpy.fea.materials import MaterialProperties


def example_default_mesher():
    """Example using default Netgen mesher (simplest approach)."""

    # Create a simple part
    sketch = Sketch("XY")
    sketch.add_rectangle(0, 0, 10, 10)
    part = sketch.extrude(5)

    # Create FEA kernel with default mesher
    kernel = TorchFEMKernel(device="cpu")  # Uses NetgenMesher by default

    print(f"Using mesher: {kernel.mesher.get_name()}")
    print(f"Supported formats: {kernel.mesher.get_supported_formats()}")
    print(f"Supported element types: {kernel.mesher.get_supported_element_types()}")


def example_explicit_netgen_mesher():
    """Example explicitly providing a Netgen mesher with custom settings."""

    # Create mesher with specific thread count
    mesher = NetgenMesher(num_threads=4)

    # Create FEA kernel with custom mesher
    kernel = TorchFEMKernel(device="cpu", mesher=mesher)

    print(f"Using mesher: {kernel.mesher.get_name()}")
    print(f"Using {kernel.mesher.num_threads} threads")


def example_check_mesher_availability():
    """Example checking if a mesher is available before using it."""

    if NetgenMesher.is_available():
        print("Netgen is available!")
        mesher = NetgenMesher(num_threads=2)
        kernel = TorchFEMKernel(device="cpu", mesher=mesher)
    else:
        print("Netgen not available, using alternative...")
        # Could try another mesher here
        pass


def example_custom_mesher():
    """
    Example showing how to create a custom mesher.

    This demonstrates the interface that any custom mesher must implement.
    """

    class CustomMesher(MesherBase):
        """Example custom mesher implementation."""

        @classmethod
        def get_name(cls) -> str:
            return "CustomMesher"

        @classmethod
        def is_available(cls) -> bool:
            # Check if dependencies are available
            return True

        def get_supported_element_types(self) -> list[str]:
            return ["tet4", "tet10", "hex8"]

        def get_supported_formats(self) -> list[str]:
            return [".step", ".stp", ".iges", ".stl"]

        def generate_mesh(
            self,
            filename,
            mesh_size=1.0,
            element_type="tet4",
            dim=3,
            verbose=True,
            **kwargs,
        ):
            """
            Your custom meshing implementation goes here.

            Should return (nodes, elements) as PyTorch tensors.
            """
            # Validate inputs first
            self.validate_inputs(filename, element_type, dim)

            # Your meshing code here...
            # nodes = ... (N, dim) torch.Tensor
            # elements = ... (M, nodes_per_element) torch.Tensor

            raise NotImplementedError("Implement your meshing logic here")

    # Use custom mesher
    # mesher = CustomMesher(num_threads=4)
    # kernel = TorchFEMKernel(device="cpu", mesher=mesher)


def example_mesher_comparison():
    """Example comparing different meshers on the same geometry."""

    import time
    from pathlib import Path

    # Create test geometry
    sketch = Sketch("XY")
    sketch.add_circle(0, 0, 5)
    part = sketch.extrude(10)

    # Export to file
    test_file = "/tmp/test_part.step"
    part.to_step(test_file)

    # Test Netgen with different thread counts
    for num_threads in [1, 2, 4]:
        mesher = NetgenMesher(num_threads=num_threads)

        t0 = time.perf_counter()
        nodes, elements = mesher.generate_mesh(
            test_file, mesh_size=1.0, element_type="tet4", verbose=False
        )
        elapsed = time.perf_counter() - t0

        print(
            f"Netgen ({num_threads} threads): "
            f"{nodes.shape[0]} nodes, {elements.shape[0]} elements, "
            f"{elapsed:.3f}s"
        )

    # Could add other meshers here for comparison
    # if GMSHMesher.is_available():
    #     mesher = GMSHMesher()
    #     ...


if __name__ == "__main__":
    print("=" * 80)
    print("Mesher Examples")
    print("=" * 80)

    print("\n1. Default Mesher:")
    print("-" * 80)
    example_default_mesher()

    print("\n2. Explicit Netgen Mesher:")
    print("-" * 80)
    example_explicit_netgen_mesher()

    print("\n3. Check Availability:")
    print("-" * 80)
    example_check_mesher_availability()

    print("\n4. Custom Mesher (see code for implementation):")
    print("-" * 80)
    print("See example_custom_mesher() function for how to implement a custom mesher")

    # Uncomment to test performance
    # print("\n5. Mesher Comparison:")
    # print("-" * 80)
    # example_mesher_comparison()
