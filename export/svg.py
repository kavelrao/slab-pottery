import numpy as np
from numpy.typing import NDArray
import svgwrite
import trimesh
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from cutting import count_boundary_loops_by_edge_index


def generate_svg(vertices_2d: NDArray[np.float32], faces: NDArray[np.int64], filename: str, cut_angles: dict[int, float] = None):
    if cut_angles is None:
        cut_angles = {}
    
    min_point = vertices_2d.min(axis=0)  # (2,)
    max_point = vertices_2d.max(axis=0)  # (2,)
    stroke_width = (max_point - min_point).min() * 0.01
    vertices_2d -= min_point
    vertices_2d += 2 * stroke_width
    size = max_point - min_point  # (2,)
    size += 8 * stroke_width

    # Add extra space at the bottom for the legend
    legend_height = size[1] * 0.2  # Reserve 20% of height for legend
    canvas_height = size[1] + legend_height
    
    svg = svgwrite.Drawing(filename, size=(size[0], canvas_height), profile='tiny')
    
    # add a zero 3rd dimension to vertices_2d
    vertices_3d = np.zeros((vertices_2d.shape[0], 3))
    vertices_3d[:, 0:2] = vertices_2d

    mesh = trimesh.Trimesh(vertices_3d, faces)
    n_boundary_loops, boundary_loop_edges = count_boundary_loops_by_edge_index(mesh)
    breakpoint()

    assert n_boundary_loops == 1
    
    # Find the range of cut angles for color mapping
    if cut_angles:
        min_angle = min(cut_angles.values())
        max_angle = max(cut_angles.values())
        angle_range = max_angle - min_angle
        
        # Create a colormap
        cmap = cm.get_cmap('viridis')
    else:
        min_angle = 0
        max_angle = 0
        angle_range = 1
    
    # Dictionary to keep track of angles we've already added to the legend
    legend_entries = {}
    
    # Draw edges with colors based on cut_angles
    for edge_idx in boundary_loop_edges[0]:
        edge = mesh.edges[edge_idx]
        v1 = vertices_2d[edge[0]]
        v2 = vertices_2d[edge[1]]

        # Check if the edge is in cut_angles
        if edge_idx in cut_angles:
            angle = cut_angles[edge_idx]
            # Normalize angle for color mapping
            if angle_range > 0:
                norm_angle = (angle - min_angle) / angle_range
            else:
                norm_angle = 0.5
            color = mcolors.to_hex(cmap(norm_angle))
            
            # Add to legend entries
            rounded_angle = round(angle)
            legend_entries[rounded_angle] = color
        else:
            color = 'black'
        
        svg.add(svg.line(start=v1, end=v2, stroke=color, stroke_width=stroke_width))
    
    # Add legend if we have cut angles
    if legend_entries:
        # Sort by angle
        sorted_angles = sorted(legend_entries.keys())
        
        # Set up legend parameters
        legend_x = stroke_width * 2
        legend_y = size[1] + stroke_width * 2
        legend_item_height = min(stroke_width * 4, legend_height / (len(legend_entries) + 2))
        legend_item_width = size[0] * 0.2
        text_offset = legend_item_width + stroke_width * 4
        
        # Add legend title
        svg.add(svg.text("Cut Angle Legend:", 
                         insert=(legend_x, legend_y),
                         font_size=legend_item_height * 1.2,
                         font_family="Arial",
                         font_weight="bold"))
        
        # Add legend items
        for i, angle in enumerate(sorted_angles):
            item_y = legend_y + (i + 1) * legend_item_height * 1.5
            color = legend_entries[angle]
            
            # Add color box
            svg.add(svg.rect(insert=(legend_x, item_y - legend_item_height * 0.8),
                           size=(legend_item_width, legend_item_height),
                           fill=color,
                           stroke="black",
                           stroke_width=stroke_width * 0.5))
            
            # Add text label
            svg.add(svg.text(f"{angle} degrees", 
                           insert=(legend_x + text_offset, item_y),
                           font_size=legend_item_height,
                           font_family="Arial"))
    
    svg.save()
