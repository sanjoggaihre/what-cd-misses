import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def generate_ply_from_labels(label_path, output_dir, output_filename):
    print(f"Reading data from: {label_path}")
    
    # Load the 4-column data: [X, Y, Z, Label]
    data = np.loadtxt(label_path)
    
    # Split into coordinates and labels
    points = data[:, :3]
    labels = data[:, 3].astype(int)
    
    # Generate distinct colors using a colormap
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    
    color_map = {}
    color_idx = 0
    for lbl_id in unique_labels:
        # Background/Noise (-1, -2) maps to light gray
        if lbl_id < 0:
            color_map[lbl_id] = [0.7, 0.7, 0.7]
        # Stem/Main body (0) maps to brown
        elif lbl_id == 0:
            color_map[lbl_id] = [0.4, 0.3, 0.2]
        # Everything else gets a bright, distinct color
        else:
            color_map[lbl_id] = cmap(color_idx % 20)[:3]
            color_idx += 1
            
    # Apply colors
    point_colors = np.array([color_map[lbl] for lbl in labels])
    
    # Export to .ply
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Successfully saved 3D visualization to: {out_path}")

if __name__ == "__main__":
    # Point directly to the label file
    label_txt = "subset_1/plant_03/labels/instance/A3_Day_00.txt"
    
    output_directory = "Instance_3D_GT_Visualization"
    output_name = "A3_Day_00_Semantic.ply"
    
    generate_ply_from_labels(label_txt, output_directory, output_name)