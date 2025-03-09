import trimesh
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from segmenting import segment_mesh_face_normals
from plotting import plot_mesh_regions, plot_mesh


def interactive_deduplicate(mesh: trimesh.Trimesh, regions):
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
    reindexed_regions: list
        List of face indices for each region, reindexed to match the new submesh
    region_selections: dict
        Dictionary with two keys:
        - 'single_regions': list of region indices selected as single regions
        - 'region_pairs': list of (outer_region, inner_region) tuples selected by the user
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox
    
    selected_region_pairs = []  # List of (outer_region, inner_region) tuples
    selected_single_regions = []  # List of single region indices
    
    # Create a matplotlib figure with 2 subplots with 3D projection
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 8))
    fig.subplots_adjust(bottom=0.2)  # Make room for the text box
    
    # Initial plot
    plot_mesh_regions(mesh, regions, title="Segmented Mesh Regions", ax=ax1)
    ax2.set_title("Selected Mesh Regions")
    
    # Add a status text at the bottom
    status_text = fig.text(0.5, 0.05, "Enter commands: s# (add single region), a#,# (add outer,inner pair), r# (remove pair), rs# (remove single), d (done)",
                          ha='center', va='center', fontsize=10)
    
    # Function to handle text input
    def submit(text):
        nonlocal selected_region_pairs, selected_single_regions
        
        action = text.strip()
        if action.lower() == 'd':
            plt.close(fig)
            return
        
        try:
            if action.lower().startswith('s'):
                # Parse the command "s#" to get the single region index
                region_idx = int(action[1:])
                
                # Validate region index
                if region_idx < 0 or region_idx >= len(regions):
                    status_text.set_text(f"Region index out of bounds. Available regions: 0-{len(regions)-1}")
                elif region_idx in selected_single_regions:
                    status_text.set_text(f"Region {region_idx} already selected.")
                else:
                    # Add the single region
                    selected_single_regions.append(region_idx)
                    status_text.set_text(f"Added single region {region_idx}. Selected singles: {selected_single_regions}")
            
            elif action.lower().startswith('a'):
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
            
            elif action.lower().startswith('rs'):
                # Remove a single region by its value
                region_idx = int(action[2:])
                if region_idx in selected_single_regions:
                    selected_single_regions.remove(region_idx)
                    status_text.set_text(f"Removed single region {region_idx}. Selected singles: {selected_single_regions}")
                else:
                    status_text.set_text(f"Region {region_idx} not in selected singles: {selected_single_regions}")
            
            elif action.lower().startswith('r'):
                # Remove a pair by its index in the selected_region_pairs list
                pair_idx = int(action[1:])
                if 0 <= pair_idx < len(selected_region_pairs):
                    removed_pair = selected_region_pairs.pop(pair_idx)
                    status_text.set_text(f"Removed pair {removed_pair}. Selected pairs: {selected_region_pairs}")
                else:
                    status_text.set_text(f"Invalid pair index. Selected pairs: {selected_region_pairs}")
            
            else:
                status_text.set_text("Invalid action. Use s# for single region, a#,# for region pair, rs# or r# to remove, or d to finish.")
        
        except ValueError:
            status_text.set_text("Invalid input. Please enter a valid command.")
        
        # Update the plots with selected regions
        ax1.clear()
        # Add region indices to visualization to clarify which number corresponds to which region
        region_labels_all = [f"Region {i}" for i in range(len(regions))]
        plot_mesh_regions(mesh, regions, title="All Mesh Regions", ax=ax1, region_labels=region_labels_all)

        # Update the visualization of selected regions (both singles and outers from pairs)
        ax2.clear()
        selected_regions_list = []
        region_labels = []
        
        # Add single regions
        for idx, region_idx in enumerate(selected_single_regions):
            selected_regions_list.append(regions[region_idx])
            region_labels.append(f"Single Region {region_idx}")
        
        # Add outer regions from pairs
        for idx, (outer_region, _) in enumerate(selected_region_pairs):
            selected_regions_list.append(regions[outer_region])
            region_labels.append(f"Outer Region {outer_region} (Pair {idx})")
        
        if selected_regions_list:
            plot_mesh_regions(mesh, selected_regions_list, 
                              title="Selected Regions", 
                              ax=ax2, 
                              region_labels=region_labels)
        else:
            ax2.set_title("No Regions Selected Yet")
        
        # Update the figure
        text_box.set_val("")  # Clear the text box
        fig.canvas.draw_idle()
    
    # Add text box for input
    ax_box = plt.axes([0.2, 0.1, 0.6, 0.05])
    text_box = TextBox(ax_box, "Command: ", initial="")
    text_box.on_submit(submit)
    
    plt.show()
    
    # Return both the mesh with selected regions and the selection information
    if selected_region_pairs or selected_single_regions:
        # Combine all selected regions (both single and outer pairs)
        selected_faces = []
        selected_regions = []
        
        # Add faces from single regions
        for region_idx in selected_single_regions:
            selected_faces.extend(regions[region_idx])
            selected_regions.append(regions[region_idx])
        
        # Add faces from outer regions in pairs
        for outer, _ in selected_region_pairs:
            selected_faces.extend(regions[outer])
            selected_regions.append(regions[outer])
        
        # Remove duplicates
        selected_faces = list(set(selected_faces))
        
        # Create submesh with selected faces
        submesh = mesh.submesh([selected_faces], append=True)
        
        # Reindex the face indices for each region based on the new submesh
        reindexed_regions = []
        
        # Create a lookup table from original mesh face index to new face index in the submesh
        # Since trimesh.submesh() doesn't have return_faces parameter in this version,
        # we need to manually build the mapping by finding each face in the new mesh
        face_index_lookup = {}
        for i, face_idx in enumerate(selected_faces):
            face_index_lookup[face_idx] = i
        
        # Reindex each selected region
        for region_faces in selected_regions:
            reindexed_region = []
            for face_idx in region_faces:
                # Check if this face is included in the new submesh
                if face_idx in face_index_lookup:
                    reindexed_region.append(face_index_lookup[face_idx])
            reindexed_regions.append(reindexed_region)
        
        # Create a dictionary to return both single regions and region pairs
        region_selections = {
            'single_regions': selected_single_regions,
            'region_pairs': selected_region_pairs
        }
        
        return submesh, reindexed_regions, region_selections
    else:
        return None, None, {'single_regions': [], 'region_pairs': []}
