import meshio
import numpy as np

def inspect_file(path, label):
    print(f"\n=== Inspecting {label} ({path}) ===")
    mesh = meshio.read(path)
    data = mesh.cell_data
    
    # Extract stress components
    sig_xx = data["sig_xx"][0]
    sig_yy = data["sig_yy"][0]
    sig_zz = data["sig_zz"][0]
    
    for key, val in data.items():
        arr = val[0]
        if "sig" in key or "eps" in key:
            print(f"  {key:<10}: mean={arr.mean():.6e}, min={arr.min():.6e}, max={arr.max():.6e}, std={arr.std():.6e}")
            
    # Calculate correlations
    corr_xx_yy = np.corrcoef(sig_xx, sig_yy)[0, 1]
    print(f"  Correlation sig_xx vs sig_yy: {corr_xx_yy:.4f}")
    if not np.allclose(sig_zz, 0.0):
        corr_xx_sig_zz = np.corrcoef(sig_xx, sig_zz)[0, 1]
        corr_yy_sig_zz = np.corrcoef(sig_yy, sig_zz)[0, 1]
        print(f"  Correlation sig_xx vs sig_zz: {corr_xx_sig_zz:.4f}")
        print(f"  Correlation sig_yy vs sig_zz: {corr_yy_sig_zz:.4f}")

inspect_file("output_plane_stress_2d_32_DAMASK_comparison/step_000500.vtu", "2D Plane Stress")
inspect_file("output_3d_32_DAMASK_comparison/step_000500.vtu", "3D nz=1")
