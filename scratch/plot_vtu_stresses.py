import meshio
import numpy as np
import matplotlib.pyplot as plt
import os

def load_vtu_slice(path):
    mesh = meshio.read(path)
    # Get grid size
    # points are mesh.points
    pts = mesh.points
    nx = len(np.unique(pts[:, 0])) - 1
    ny = len(np.unique(pts[:, 1])) - 1
    
    sig_xx = mesh.cell_data["sig_xx"][0].reshape(nx, ny)
    sig_yy = mesh.cell_data["sig_yy"][0].reshape(nx, ny)
    sig_zz = mesh.cell_data["sig_zz"][0].reshape(nx, ny)
    
    return sig_xx, sig_yy, sig_zz

sig_xx_2d, sig_yy_2d, sig_zz_2d = load_vtu_slice("output_plane_stress_2d_32_DAMASK_comparison/step_000500.vtu")
sig_xx_3d, sig_yy_3d, sig_zz_3d = load_vtu_slice("output_3d_32_DAMASK_comparison/step_000500.vtu")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 2D Plotting
im1 = axes[0, 0].imshow(sig_xx_2d.T, origin='lower', cmap='viridis')
axes[0, 0].set_title("2D Plane Stress: sig_xx")
fig.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(sig_yy_2d.T, origin='lower', cmap='viridis')
axes[0, 1].set_title("2D Plane Stress: sig_yy")
fig.colorbar(im2, ax=axes[0, 1])

im3 = axes[0, 2].imshow(sig_zz_2d.T, origin='lower', cmap='viridis')
axes[0, 2].set_title("2D Plane Stress: sig_zz")
fig.colorbar(im3, ax=axes[0, 2])

# 3D Plotting
im4 = axes[1, 0].imshow(sig_xx_3d.T, origin='lower', cmap='viridis')
axes[1, 0].set_title("3D nz=1: sig_xx")
fig.colorbar(im4, ax=axes[1, 0])

im5 = axes[1, 1].imshow(sig_yy_3d.T, origin='lower', cmap='viridis')
axes[1, 1].set_title("3D nz=1: sig_yy")
fig.colorbar(im5, ax=axes[1, 1])

im6 = axes[1, 2].imshow(sig_zz_3d.T, origin='lower', cmap='viridis')
axes[1, 2].set_title("3D nz=1: sig_zz")
fig.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
out_path = "C:/Users/siama/.gemini/antigravity-ide/brain/07ee13ce-a58d-44b0-a82c-6f24a1fd347f/stress_patterns_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"Plot saved successfully to {out_path}")
