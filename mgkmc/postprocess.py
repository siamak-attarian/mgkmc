import numpy as np
import meshio

def reconstruct_displacement_element_centered(eps, pixel=1.0):
    """
    Element-centered displacement reconstruction for visualization.

    Modified so that:
    - The left boundary (x=0) is the fixed visual anchor.
    - Tension expands only toward +x.
    - First element center is at (pixel/2, pixel/2, pixel/2).
    """

    nx, ny, nz = eps.shape[:3]
    Nx, Ny, Nz = nx+1, ny+1, nz+1

    u = np.zeros((Nx,Ny,Nz,3))

    # --------------------------------
    # Integrate NORMAL strains
    # --------------------------------

    # u_x from eps_xx: integrate along +x direction
    for i in range(1, Nx):
        ix = i - 1
        for j in range(Ny):
            jy = min(j, ny-1)
            for k in range(Nz):
                kz = min(k, nz-1)
                u[i,j,k,0] = u[i-1,j,k,0] + eps[ix,jy,kz,0,0] * pixel

    # u_y from eps_yy
    for j in range(1, Ny):
        jy = j - 1
        for i in range(Nx):
            ix = min(i, nx-1)
            for k in range(Nz):
                kz = min(k, nz-1)
                u[i,j,k,1] = u[i,j-1,k,1] + eps[ix,jy,kz,1,1] * pixel

    # u_z from eps_zz
    for k in range(1, Nz):
        kz = k - 1
        for i in range(Nx):
            ix = min(i, nx-1)
            for j in range(Ny):
                jy = min(j, ny-1)
                u[i,j,k,2] = u[i,j,k-1,2] + eps[ix,jy,kz,2,2] * pixel

    # --------------------------------
    # SHEAR contributions: γ = 2 ε_xy
    # --------------------------------
    g_xy = eps[:,:,:,0,1]
    g_xz = eps[:,:,:,0,2]
    g_yz = eps[:,:,:,1,2]

    # u_x from shear
    for j in range(1, Ny):
        for i in range(Nx):
            ix = min(i, nx-1)
            for k in range(Nz):
                kz = min(k, nz-1)
                u[i,j,k,0] += 0.5 * g_xy[ix,j-1,kz] * pixel

    for k in range(1, Nz):
        for i in range(Nx):
            ix = min(i, nx-1)
            for j in range(Ny):
                jy = min(j, ny-1)
                u[i,j,k,0] += 0.5 * g_xz[ix,jy,k-1] * pixel

    # u_y from shear
    for i in range(1, Nx):
        for j in range(Ny):
            jy = min(j, ny-1)
            for k in range(Nz):
                kz = min(k, nz-1)
                u[i,j,k,1] += 0.5 * g_xy[i-1,jy,kz] * pixel

    for k in range(1, Nz):
        for i in range(Nx):
            ix = min(i, nx-1)
            for j in range(Ny):
                jy = min(j, ny-1)
                u[i,j,k,1] += 0.5 * g_yz[ix,jy,k-1] * pixel

    # u_z from shear
    for i in range(1, Nx):
        for j in range(Ny):
            jy = min(j, ny-1)
            for k in range(Nz):
                kz = min(k, nz-1)
                u[i,j,k,2] += 0.5 * g_xz[i-1,jy,kz] * pixel

    for j in range(1, Ny):
        for i in range(Nx):
            ix = min(i, nx-1)
            for k in range(Nz):
                kz = min(k, nz-1)
                u[i,j,k,2] += 0.5 * g_yz[ix,j-1,kz] * pixel

    # --------------------------------
    # FIX VISUAL ANCHOR: left boundary (x=0)
    # --------------------------------
    # Shift so that ALL nodes at x=0 are exactly at zero displacement
    u0 = u[0,:,:,:].mean(axis=(0,1))
    u -= u0.reshape(1,1,1,3)

    return u


def export_to_vtk(filename, eps, sig, E, nu, pixel=1.0,
                  match_matplotlib_orientation=False):
    """
    Export voxel-based FFT results to VTK.

    Parameters
    ----------
    match_matplotlib_orientation : bool
        If True, the VTK view will look exactly like matplotlib imshow(origin='lower').
    """

    nx, ny, nz = E.shape

    # ---------------------------
    # 2. Coordinates (nx+1, ny+1, nz+1)
    # ---------------------------
    x = np.arange(nx+1) * pixel
    y = np.arange(ny+1) * pixel
    z = np.arange(nz+1) * pixel

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # ---------------------------
    # 3. Hex connectivity
    # ---------------------------
    cells = []
    NYZ = (ny+1)*(nz+1)
    NZ  = nz+1

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = i*NYZ + j*NZ + k
                n1 = n0 + 1
                n2 = n0 + NZ
                n3 = n2 + 1
                n4 = n0 + NYZ
                n5 = n4 + 1
                n6 = n4 + NZ
                n7 = n6 + 1
                cells.append([n0,n1,n3,n2,n4,n5,n7,n6])

    cells = [("hexahedron", np.array(cells, int))]

 # ---- Von Mises stress ----
    tr_sig = np.trace(sig, axis1=3, axis2=4)[..., None, None]
    sig_dev = sig - np.eye(3)[None,None,None,:,:] * tr_sig/3
    sig_vm = np.sqrt(1.5 * np.sum(sig_dev**2, axis=(3,4)))

    # ---- Von Mises strain ----
    tr_eps = np.trace(eps, axis1=3, axis2=4)[..., None, None]
    eps_dev = eps - np.eye(3)[None,None,None,:,:] * tr_eps/3
    eps_vm = np.sqrt((2/3) * np.sum(eps_dev**2, axis=(3,4)))

    # ---- Engineering shear strains ----
    eps_xy = eps[...,0,1]
    eps_xz = eps[...,0,2]
    eps_yz = eps[...,1,2]

    # ---- Shear stresses ----
    sig_xy = sig[...,0,1]
    sig_xz = sig[...,0,2]
    sig_yz = sig[...,1,2]

    # ---------------------------
    # Cell data
    # ---------------------------
    cell_data = {
        "E"       : [E.ravel(order="C")],
        "nu"      : [nu.ravel(order="C")],

        # Normal strains
        "eps_xx"  : [eps[...,0,0].ravel(order="C")],
        "eps_yy"  : [eps[...,1,1].ravel(order="C")],
        "eps_zz"  : [eps[...,2,2].ravel(order="C")],

        # Shear strains (Tensorial)
        "eps_xy"  : [eps_xy.ravel(order="C")],
        "eps_xz"  : [eps_xz.ravel(order="C")],
        "eps_yz"  : [eps_yz.ravel(order="C")],

        # Normal stresses
        "sig_xx"  : [sig[...,0,0].ravel(order="C")],
        "sig_yy"  : [sig[...,1,1].ravel(order="C")],
        "sig_zz"  : [sig[...,2,2].ravel(order="C")],

        # Shear stresses
        "sig_xy"  : [sig_xy.ravel(order="C")],
        "sig_xz"  : [sig_xz.ravel(order="C")],
        "sig_yz"  : [sig_yz.ravel(order="C")],

        # Invariant measures
        "sig_vm"  : [sig_vm.ravel(order="C")],
        "eps_vm"  : [eps_vm.ravel(order="C")],
    }

    # ---------------------------
    # 6. Optional displacement
    # ---------------------------
    u_nodes = reconstruct_displacement_element_centered(eps, pixel=pixel)
    point_data = {"displacement": u_nodes.reshape(-1,3)}

    # ---------------------------
    # 7. Write VTK
    # ---------------------------
    mesh = meshio.Mesh(
        points=points,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data
    )
    mesh.write(filename)
    #print(f"VTK export complete: {filename}")


def export_simulation_vtk(
    eps_list,
    sig_list,
    E, nu,
    pixel=1.0,
    steps="last",       # "last", "all", or [list_of_steps]
    prefix="step_"
):
    """
    Export VTK files for a completed simulation.
    
    Parameters
    ----------
    eps_list, sig_list : list of ndarray
        Lists of element-centered strain/stress fields for each step.
    E, nu : ndarray
        Material fields (element-centered)
    pixel : float
        Element size
    steps : str or list
        "last"  -> export only the final step
        "all"   -> export all steps
        [list]  -> export selected steps (e.g. [0,5,10])
    prefix : str
        Filename prefix for the VTK files.
    """
    
    n_steps = len(eps_list)

    if steps == "last":
        idx = n_steps - 1
        fname = f"{prefix}final.vtu"
        export_to_vtk(fname, eps_list[idx], sig_list[idx], E, nu, pixel,
                      match_matplotlib_orientation=True)
        return

    if steps == "all":
        for i in range(n_steps):
            fname = f"{prefix}{i:03d}.vtu"
            export_to_vtk(fname, eps_list[i], sig_list[i], E, nu, pixel,
                          match_matplotlib_orientation=True)
        return

    if isinstance(steps, (list, tuple)):
        for i in steps:
            if 0 <= i < n_steps:
                fname = f"{prefix}{i:03d}.vtu"
                export_to_vtk(fname, eps_list[i], sig_list[i], E, nu, pixel,
                              match_matplotlib_orientation=True)
        return

    raise ValueError("steps must be 'last', 'all', or a list of integers.")
