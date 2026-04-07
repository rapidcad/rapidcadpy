"""
Visualization helpers for boundary conditions.
"""


def visualize_boundary_conditions(
    model, nodes, elements, window_size=(1400, 700), interactive=True
):
    """
    Visualize boundary conditions (constraints and loads) on a mesh.

    This function creates an interactive 3D visualization showing:
    - The mesh in semi-transparent blue
    - Fixed/constrained nodes as red spheres
    - Loaded nodes as green spheres
    - Force vectors as green arrows

    Args:
        model: FEA solver model object with constraints and forces attributes
        nodes: Mesh nodes tensor (n_nodes, 3)
        elements: Mesh elements tensor (n_elements, nodes_per_element)
        window_size: Window size as (width, height). Default: (1400, 700)
        interactive: If True open an interactive window; False renders off-screen.

    Returns:
        pyvista.Plotter: Configured plotter (call .show() to display).

    Example:
        >>> from rapidcadpy.fea.boundary_conditions import visualize_boundary_conditions
        >>> visualize_boundary_conditions(model, nodes, elements)
    """
    try:
        import pyvista as pv
        import numpy as np
    except ImportError:
        raise ImportError(
            "PyVista is required for boundary condition visualization. "
            "Install it with: pip install pyvista"
        )

    # Create PyVista mesh for visualization
    points = nodes.cpu().numpy()
    cells = elements.cpu().numpy()

    # Create VTK cells (prepend count of nodes per element)
    nodes_per_elem = cells.shape[1]
    vtk_cells = np.column_stack([np.full(len(cells), nodes_per_elem), cells]).ravel()

    # Determine cell type based on nodes per element
    if nodes_per_elem == 4:
        celltypes = np.full(len(cells), pv.CellType.TETRA)
    elif nodes_per_elem == 8:
        celltypes = np.full(len(cells), pv.CellType.HEXAHEDRON)
    elif nodes_per_elem == 10:
        celltypes = np.full(len(cells), pv.CellType.QUADRATIC_TETRA)
    elif nodes_per_elem == 20:
        celltypes = np.full(len(cells), pv.CellType.QUADRATIC_HEXAHEDRON)
    else:
        raise ValueError(f"Unsupported element type with {nodes_per_elem} nodes")

    pv_mesh = pv.UnstructuredGrid(vtk_cells, celltypes, points)

    # Create plotter - use off_screen if saving to file
    off_screen = not interactive
    plotter = pv.Plotter(window_size=window_size, off_screen=off_screen)

    # Add the main mesh (semi-transparent)
    plotter.add_mesh(
        pv_mesh,
        color="lightblue",
        opacity=0.3,
        show_edges=True,
        edge_color="gray",
        line_width=0.5,
    )

    # Visualize FIXED NODES (constraints)
    constrained_mask = model.constraints.any(dim=1).cpu().numpy()
    constrained_nodes = nodes[constrained_mask].cpu().numpy()

    if len(constrained_nodes) > 0:
        fixed_points = pv.PolyData(constrained_nodes)
        plotter.add_mesh(
            fixed_points,
            color="red",
            point_size=15,
            render_points_as_spheres=True,
            label="Fixed Nodes",
        )
        print(f"✓ Visualizing {len(constrained_nodes)} constrained nodes (RED)")

    # Visualize LOADED NODES (forces)
    force_mask = (model.forces.abs() > 1e-10).any(dim=1).cpu().numpy()
    loaded_nodes = nodes[force_mask].cpu().numpy()
    force_vectors = model.forces[force_mask].cpu().numpy()

    if len(loaded_nodes) > 0:
        load_points = pv.PolyData(loaded_nodes)
        plotter.add_mesh(
            load_points,
            color="green",
            point_size=15,
            render_points_as_spheres=True,
            label="Loaded Nodes",
        )

        arrow_scale = (nodes[:, 0].max() - nodes[:, 0].min()).item() * 0.1
        force_magnitude = np.linalg.norm(force_vectors, axis=1, keepdims=True)
        force_directions = force_vectors / (force_magnitude + 1e-10)
        scaled_vectors = force_directions * arrow_scale

        plotter.add_arrows(
            loaded_nodes,
            scaled_vectors,
            mag=1.0,
            color="darkgreen",
            label="Force Vectors",
        )
        print(
            f"✓ Visualizing {len(loaded_nodes)} loaded nodes (GREEN) with force arrows"
        )

    plotter.add_legend()
    plotter.add_text("Boundary Conditions", position="upper_edge", font_size=12)
    plotter.add_text(
        "Red = Fixed (Constraints)\nGreen = Loaded (Forces)",
        position="lower_left",
        font_size=10,
    )
    plotter.add_axes()
    plotter.camera_position = "iso"

    return plotter
