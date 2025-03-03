import trimesh
import matplotlib.pyplot as plt
from pathlib import Path

from segmenting import segment_mesh_face_normals
from plotting import plot_mesh_regions, plot_mesh


def interactive_deduplicate(mesh: trimesh.Trimesh, segment_fn=segment_mesh_face_normals):
    """
    Interactive deduplication of a mesh using face normal segmentation.

    Parameters:
    ----------
    mesh : trimesh.Trimesh
        The input mesh to deduplicate
    segment_fn : callable mesh -> list[NDArray[np.int64]]
        A function that segments the input mesh into regions based on some criteria.
        The function should return a list of face indices for each region.
        
    Returns:
    -------
    trimesh.Trimesh
        A new mesh with only the selected regions
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox
    
    regions = segment_fn(mesh)
    selected_regions = []
    
    # Create a matplotlib figure with 2 subplots with 3D projection
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.2)  # Make room for the text box
    
    # Initial plot
    plot_mesh_regions(mesh, regions, title="Segmented Mesh Regions", ax=ax1)
    ax2.set_title("Selected Mesh Regions")
    
    # Add a status text at the bottom
    status_text = fig.text(0.5, 0.05, "Enter commands: a# (add region), r# (remove region), d (done)",
                          ha='center', va='center', fontsize=10)
    
    # Function to handle text input
    def submit(text):
        nonlocal selected_regions
        
        action = text.strip()
        if action.lower() == 'd':
            plt.close(fig)
            return
        
        try:
            if action.lower().startswith('a'):
                region_idx = int(action[1:])
                if region_idx < 0 or region_idx >= len(regions):
                    status_text.set_text(f"Region index {region_idx} out of bounds. Available regions: 0-{len(regions)-1}")
                elif region_idx not in selected_regions:
                    selected_regions.append(region_idx)
                    status_text.set_text(f"Added region {region_idx}. Selected regions: {selected_regions}")
            elif action.lower().startswith('r'):
                region_idx = int(action[1:])
                if region_idx in selected_regions:
                    selected_regions.remove(region_idx)
                    status_text.set_text(f"Removed region {region_idx}. Selected regions: {selected_regions}")
                else:
                    status_text.set_text(f"Region {region_idx} not in selected regions: {selected_regions}")
            else:
                status_text.set_text("Invalid action. Use a# to add region, r# to remove region, or d to finish.")
        except ValueError:
            status_text.set_text("Invalid input. Please enter a valid command.")
        
        # Update the plots with selected regions
        ax1.clear()
        unselected_regions, unselected_region_labels = zip(*[(region, f"Region {i}") for i, region in enumerate(regions) if i not in selected_regions])
        plot_mesh_regions(mesh, unselected_regions, title="Unselected Mesh Regions", ax=ax1, region_labels=unselected_region_labels)

        ax2.clear()
        if selected_regions:
            region_labels = [f"Region {i}" for i in selected_regions]
            plot_mesh_regions(mesh, [regions[idx] for idx in selected_regions], title="Selected Mesh Regions", ax=ax2, region_labels=region_labels)
        
        # Update the figure
        text_box.set_val("")  # Clear the text box
        fig.canvas.draw_idle()
    
    # Add text box for input
    ax_box = plt.axes([0.2, 0.1, 0.6, 0.05])
    text_box = TextBox(ax_box, "Command: ", initial="")
    text_box.on_submit(submit)
    
    plt.show()
    
    # Return the mesh with selected regions after the interactive session is done
    if selected_regions:
        # Combine the selected regions
        selected_faces = []
        for region_idx in selected_regions:
            selected_faces.extend(regions[region_idx])
        
        return mesh.submesh([selected_faces], append=True)
    else:
        return None


if  __name__ == '__main__':
    filename = "Mug_Thick_Handle"
    og_mesh = trimesh.load(Path(__file__).parent.parent / "files" / f"{filename}.stl")
    mesh = interactive_deduplicate(og_mesh, segment_fn=lambda mesh: segment_mesh_face_normals(mesh, angle_threshold=30))

    if mesh is not None:
        with open(Path(__file__).parent.parent / "files" / f"{filename}_Selected.stl", "wb") as f:
            mesh.export(f, file_type="stl")

        plot_mesh(mesh, title="Selected Mesh")
        plt.show()
