import numpy as np
import pytest
import os
import tempfile
from mgkmc.kmc_simulator import KmcSimulation2D
from mgkmc.aqs import ThermalSimulation

def test_thermal_conduction_2d():
    nx, ny = 16, 16
    E_field = np.full((nx, ny), 70.0 * 1e9)
    nu_field = np.full((nx, ny), 0.3)
    
    sim = KmcSimulation2D(
        nx=nx, ny=ny, M=20, gamma0=0.14,
        E_field=E_field, nu_field=nu_field,
        enable_thermal=True, Cp=420.0, rho=6125.0,
        thermal_diffusivity=3.0e-6, thermal_coords="pixel",
        temperature=300.0
    )
    
    # Initialize a temperature spike at the center
    sim.Tlocal = np.full((nx, ny), 300.0, dtype=np.float64)
    cx, cy = nx // 2, ny // 2
    sim.Tlocal[cx, cy] = 500.0
    
    initial_sum = np.sum(sim.Tlocal)
    
    # Run conduction step
    dt = 1e-3
    sim.heat_conducting_2d(dt)
    
    # 1. Total energy (temperature sum) must be conserved
    final_sum = np.sum(sim.Tlocal)
    assert np.allclose(initial_sum, final_sum, rtol=1e-5), "Energy not conserved in 2D conduction!"
    
    # 2. Spike voxel temperature must decrease
    assert sim.Tlocal[cx, cy] < 500.0, "Spike temperature did not decay!"
    
    # 3. Neighboring voxels temperature must increase
    assert sim.Tlocal[cx + 1, cy] > 300.0, "Heat did not diffuse to neighbor!"
    assert sim.Tlocal[cx - 1, cy] > 300.0, "Heat did not diffuse to neighbor!"
    assert sim.Tlocal[cx, cy + 1] > 300.0, "Heat did not diffuse to neighbor!"
    assert sim.Tlocal[cx, cy - 1] > 300.0, "Heat did not diffuse to neighbor!"


def test_thermal_conduction_3d():
    nx, ny, nz = 8, 8, 8
    E_field = np.full((nx, ny, nz), 70.0 * 1e9)
    nu_field = np.full((nx, ny, nz), 0.3)
    
    sim = ThermalSimulation(
        nx=nx, ny=ny, nz=nz, M=20, gamma0=0.14,
        E_field=E_field, nu_field=nu_field,
        enable_thermal=True, Cp=420.0, rho=6125.0,
        thermal_diffusivity=3.0e-6, thermal_coords="pixel",
        temperature=300.0
    )
    
    # Initialize a temperature spike at the center
    sim.Tlocal = np.full((nx, ny, nz), 300.0, dtype=np.float64)
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    sim.Tlocal[cx, cy, cz] = 500.0
    
    initial_sum = np.sum(sim.Tlocal)
    
    # Run conduction step
    dt = 1e-3
    sim.heat_conducting_3d(dt)
    
    # 1. Total energy (temperature sum) must be conserved
    final_sum = np.sum(sim.Tlocal)
    assert np.allclose(initial_sum, final_sum, rtol=1e-5), "Energy not conserved in 3D conduction!"
    
    # 2. Spike voxel temperature must decrease
    assert sim.Tlocal[cx, cy, cz] < 500.0, "Spike temperature did not decay!"
    
    # 3. Neighboring voxels temperature must increase
    assert sim.Tlocal[cx + 1, cy, cz] > 300.0, "Heat did not diffuse to neighbor!"
    assert sim.Tlocal[cx, cy + 1, cz] > 300.0, "Heat did not diffuse to neighbor!"
    assert sim.Tlocal[cx, cy, cz + 1] > 300.0, "Heat did not diffuse to neighbor!"


def test_thermal_checkpoint_3d():
    nx, ny, nz = 4, 4, 4
    E_field = np.full((nx, ny, nz), 70.0 * 1e9)
    nu_field = np.full((nx, ny, nz), 0.3)
    
    sim = ThermalSimulation(
        nx=nx, ny=ny, nz=nz, M=20, gamma0=0.14,
        E_field=E_field, nu_field=nu_field,
        enable_thermal=True, Cp=420.0, rho=6125.0,
        thermal_diffusivity=3.0e-6, thermal_coords="pixel",
        temperature=300.0, temperature_cap=1200.0
    )
    
    # Modify Tlocal to have some non-uniform temperature field
    sim.Tlocal[1, 1, 1] = 450.0
    sim.Tlocal[2, 2, 2] = 600.0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_thermal_cp.h5")
        sim.save_checkpoint(checkpoint_path, step=42)
        
        # Load simulation back
        sim_loaded = ThermalSimulation.load_checkpoint(checkpoint_path)
        
        # Verify all metadata and arrays loaded correctly
        assert sim_loaded.enable_thermal == sim.enable_thermal
        assert sim_loaded.Cp == sim.Cp
        assert sim_loaded.rho == sim.rho
        assert sim_loaded.thermal_diffusivity == sim.thermal_diffusivity
        assert sim_loaded.thermal_coords == sim.thermal_coords
        assert sim_loaded.temperature_cap == sim.temperature_cap
        
        assert np.allclose(sim_loaded.Tlocal, sim.Tlocal)


def test_thermostat_2d():
    nx, ny = 8, 8
    E_field = np.full((nx, ny), 70.0 * 1e9)
    nu_field = np.full((nx, ny), 0.3)
    
    # 1. Test hard rescaling (tau_bath = 0.0)
    sim = KmcSimulation2D(
        nx=nx, ny=ny, M=20, gamma0=0.14,
        E_field=E_field, nu_field=nu_field,
        enable_thermal=True, Cp=420.0, rho=6125.0,
        thermal_diffusivity=3.0e-6, thermal_coords="pixel",
        temperature=300.0, thermostat=True, tau_bath=0.0
    )
    
    # Give a temperature spike
    sim.Tlocal[3, 3] = 600.0
    assert np.mean(sim.Tlocal) > 300.0
    
    sim.heat_conducting_2d(1e-3)
    # The average temperature must be exactly reset to 300.0 K
    assert np.allclose(np.mean(sim.Tlocal), 300.0, atol=1e-6)
    
    # 2. Test relaxation (tau_bath > 0)
    sim_relax = KmcSimulation2D(
        nx=nx, ny=ny, M=20, gamma0=0.14,
        E_field=E_field, nu_field=nu_field,
        enable_thermal=True, Cp=420.0, rho=6125.0,
        thermal_diffusivity=3.0e-6, thermal_coords="pixel",
        temperature=300.0, thermostat=True, tau_bath=1e-3
    )
    
    sim_relax.Tlocal = np.full((nx, ny), 400.0, dtype=np.float64)
    dt = 1e-3
    sim_relax.heat_conducting_2d(dt)
    
    # T_expected = Tambient + (T_initial - Tambient) * exp(-dt / tau_bath)
    # T_expected = 300 + (400 - 300) * exp(-1) = 300 + 100 * exp(-1) = 336.7879 K
    expected_mean = 300.0 + 100.0 * np.exp(-1.0)
    assert np.allclose(np.mean(sim_relax.Tlocal), expected_mean, rtol=1e-4)

