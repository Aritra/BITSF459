import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# ---------------------- User Editable Transformation Parameters ----------------------
# Scale factors along x, y, z axes
Sx, Sy, Sz = 1.0, 1.0, 1.0  # Scale factors

# Translation factors along x, y, z axes
Tx, Ty, Tz = 0.0, 0.0, 0.0  # Translation factors

# Rotation angles (in degrees) about x, y, z axes
Rx, Ry, Rz = 0.0, 0.0, 0.0  # Rotation factors
# -------------------------------------------------------------------------------------

def apply_transform(geometry):
    # Build transformation matrix: scale -> rotate -> translate
    # 1. Scaling matrix (non-uniform)
    S = np.diag([Sx, Sy, Sz, 1.0])

    # 2. Rotation matrix (from Euler angles)
    rot = R.from_euler('xyz', [Rx, Ry, Rz], degrees=True)
    R_mat = np.eye(4)
    R_mat[:3, :3] = rot.as_matrix()

    # 3. Translation matrix
    T = np.eye(4)
    T[:3, 3] = [Tx, Ty, Tz]

    # Combine: T * R * S
    transform = T @ R_mat @ S

    geometry.transform(transform)
    return geometry

def main():
    parser = argparse.ArgumentParser(description="Open3D PLY Viewer with Modes and Transformations")
    parser.add_argument("ply_path", type=str, help="Path to the PLY file")
    parser.add_argument("--mode", type=str, choices=["pc", "mesh", "surface"], default="pc",
                        help="Visualization mode: 'pc' (point cloud), 'mesh' (wireframe), 'surface' (surface rendered)")
    args = parser.parse_args()

    # Load geometry
    mesh = o3d.io.read_triangle_mesh(args.ply_path)
    if not mesh.has_vertices():
        print("Error: The PLY file does not contain valid mesh data.")
        return

    # Apply transformation
    mesh = apply_transform(mesh)

    if args.mode == "pc":
        # Convert mesh to point cloud
        pcd = mesh.sample_points_uniformly(number_of_points=50000)
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud View")
    elif args.mode == "mesh":
        # Render mesh as wireframe
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        o3d.visualization.draw_geometries([mesh], window_name="Mesh (Wireframe) View", mesh_show_wireframe=True)
    elif args.mode == "surface":
        # Render mesh with surface shading
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        o3d.visualization.draw_geometries([mesh], window_name="Surface Rendered View", mesh_show_wireframe=False)

if __name__ == "__main__":
    main()