import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

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
    region_pairs: list of tuples
        List of (outer_region, inner_region) pairs selected by the user
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox
    
    regions = segment_fn(mesh)
    selected_region_pairs = []  # List of (outer_region, inner_region) tuples
    
    # Create a matplotlib figure with 2 subplots with 3D projection
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.2)  # Make room for the text box
    
    # Initial plot
    plot_mesh_regions(mesh, regions, title="Segmented Mesh Regions", ax=ax1)
    ax2.set_title("Selected Mesh Regions (Outer Only)")
    
    # Add a status text at the bottom
    status_text = fig.text(0.5, 0.05, "Enter commands: a#,# (add outer,inner pair), r# (remove pair index), d (done)",
                          ha='center', va='center', fontsize=10)
    
    # Debug text to show region count
    debug_text = fig.text(0.5, 0.02, f"Available regions: 0-{len(regions)-1}", 
                         ha='center', va='center', fontsize=9)
    
    # Function to handle text input
    def submit(text):
        nonlocal selected_region_pairs
        
        action = text.strip()
        if action.lower() == 'd':
            plt.close(fig)
            return
        
        try:
            if action.lower().startswith('a'):
                # Parse the command "a#,#" to get the outer and inner region indices
                region_indices = action[1:].split(',')
                if len(region_indices) != 2:
                    status_text.set_text("Invalid format. Use a#,# to add an outer,inner region pair.")
                    return
                
                outer_region = int(region_indices[0])
                inner_region = int(region_indices[1])

                # Validate region indices
                if outer_region < 0 or outer_region >= len(regions) or inner_region < 0 or inner_region >= len(regions):
                    status_text.set_text(f"Region index out of bounds. Available regions: 0-{len(regions)-1}")
                elif (outer_region, inner_region) in selected_region_pairs:
                    status_text.set_text(f"Region pair ({outer_region},{inner_region}) already selected.")
                else:
                    # Add the pair
                    selected_region_pairs.append((outer_region, inner_region))
                    status_text.set_text(f"Added region pair ({outer_region},{inner_region}). Selected pairs: {selected_region_pairs}")
            
            elif action.lower().startswith('r'):
                # Remove a pair by its index in the selected_region_pairs list
                pair_idx = int(action[1:])
                if 0 <= pair_idx < len(selected_region_pairs):
                    removed_pair = selected_region_pairs.pop(pair_idx)
                    status_text.set_text(f"Removed pair {removed_pair}. Selected pairs: {selected_region_pairs}")
                else:
                    status_text.set_text(f"Invalid pair index. Selected pairs: {selected_region_pairs}")
            
            else:
                status_text.set_text("Invalid action. Use a#,# to add region pair, r# to remove pair, or d to finish.")
        
        except ValueError:
            status_text.set_text("Invalid input. Please enter a valid command.")
        
        # Update the plots with selected regions
        ax1.clear()
        # Add region indices to visualization to clarify which number corresponds to which region
        region_labels_all = [f"Region {i}" for i in range(len(regions))]
        plot_mesh_regions(mesh, regions, title="All Mesh Regions", ax=ax1, region_labels=region_labels_all)

        # Update the visualization of selected pairs - OUTER REGIONS ONLY
        ax2.clear()
        if selected_region_pairs:
            # Only get outer regions from the pairs
            outer_regions_indices = [outer for outer, _ in selected_region_pairs]
            
            # Create labels for the visualization
            region_labels = []
            outer_regions_list = []
            
            for idx, region_idx in enumerate(outer_regions_indices):
                outer_regions_list.append(regions[region_idx])
                
                # Create a descriptive label showing which pair this is the outer region for
                label = f"Outer Region {region_idx} (Pair {idx})"
                region_labels.append(label)
            
            plot_mesh_regions(mesh, outer_regions_list, 
                             title="Selected Outer Regions", 
                             ax=ax2, 
                             region_labels=region_labels)
        else:
            ax2.set_title("No Outer Regions Selected Yet")
        
        # Update the figure
        text_box.set_val("")  # Clear the text box
        fig.canvas.draw_idle()
    
    # Add text box for input
    ax_box = plt.axes([0.2, 0.1, 0.6, 0.05])
    text_box = TextBox(ax_box, "Command: ", initial="")
    text_box.on_submit(submit)
    
    plt.show()
    
    # Return both the mesh with selected regions and the region pairs
    if selected_region_pairs:
        # Combine all selected regions (both outer and inner)
        selected_faces = []
        for outer, _ in selected_region_pairs:
            selected_faces.extend(regions[outer])
        
        # Remove duplicates
        selected_faces = list(set(selected_faces))
        
        return mesh.submesh([selected_faces], append=True), selected_region_pairs
    else:
        return None, []


if __name__ == '__main__':
    filename = "Mug_Thick_Handle"
    og_mesh = trimesh.load(Path(__file__).parent.parent / "files" / f"{filename}.stl")
    mesh, region_pairs = interactive_deduplicate(og_mesh, segment_fn=lambda mesh: segment_mesh_face_normals(mesh, angle_threshold=30))

    if mesh is not None:
        # Export the selected mesh
        with open(Path(__file__).parent.parent / "files" / f"{filename}_Selected.stl", "wb") as f:
            mesh.export(f, file_type="stl")
        
        # Print the selected region pairs for reference
        print(f"Selected region pairs (outer, inner):")
        for i, (outer, inner) in enumerate(region_pairs):
            print(f"  Pair {i}: Outer={outer}, Inner={inner}")

        plot_mesh(mesh, title="Selected Mesh")
        plt.show()
