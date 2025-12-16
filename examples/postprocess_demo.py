"""
Demonstration of post-processing tools for simulation history analysis.

This script shows how to:
1. Load simulation history from output directory
2. Analyze cascade statistics
3. Extract stress-strain curves
4. Analyze plastic strain evolution from checkpoint
5. Generate summary report
"""

import numpy as np
import os
from mgkmc.postprocess_history import (
    extract_history,
    analyze_cascades,
    extract_stress_strain_curves,
    compute_plastic_strain_evolution,
    generate_summary_report
)

def main():
    print("=" * 60)
    print("POST-PROCESSING DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Configuration
    # You can change these to point to your simulation output
    OUTPUT_DIR = "mgkmc/examples/aqs_demo_output_uniaxial_128_new_2000_log(0.4)_5"
    CHECKPOINT_FILE = None  # Set to checkpoint file path if available
    
    # Check if output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Output directory '{OUTPUT_DIR}' not found.")
        print("Please run a simulation first (e.g., aqs_demo_2.py) or update OUTPUT_DIR")
        return
    
    print(f"Analyzing simulation output from: {OUTPUT_DIR}")
    print()
    
    # ========================================
    # 1. Extract History
    # ========================================
    print("1. Extracting simulation history...")
    print("-" * 60)
    
    hist = extract_history(OUTPUT_DIR)
    
    if hist['global'] is not None:
        n_steps = len(hist['global'])
        print(f"[OK] Loaded global history: {n_steps} steps")
        print(f"  Columns: step, eps (6), sig (6), cascade_steps, total_flips")
    else:
        print("[FAIL] Could not load global history")
    
    if hist['cascade']:
        n_cascades = len(hist['cascade'])
        print(f"[OK] Loaded cascade history: {n_cascades} cascade events")
    else:
        print("[WARN] No cascade data found")
    
    print()
    
    # ========================================
    # 2. Analyze Cascades
    # ========================================
    print("2. Analyzing cascade statistics...")
    print("-" * 60)
    
    stats = analyze_cascades(OUTPUT_DIR)
    
    print(f"Total cascade events: {stats['total_cascades']}")
    if stats['total_cascades'] > 0:
        print(f"Mean cascade size: {stats['mean_cascade_size']:.2f} flips")
        print(f"Max cascade size: {stats['max_cascade_size']} flips")
        print(f"Total flips across all steps: {stats['total_flips_per_step'].sum()}")
        
        # Show distribution
        if len(stats['cascade_sizes']) > 0:
            print(f"\nCascade size distribution:")
            print(f"  Min: {stats['cascade_sizes'].min()}")
            print(f"  25th percentile: {np.percentile(stats['cascade_sizes'], 25):.0f}")
            print(f"  Median: {np.median(stats['cascade_sizes']):.0f}")
            print(f"  75th percentile: {np.percentile(stats['cascade_sizes'], 75):.0f}")
            print(f"  Max: {stats['cascade_sizes'].max()}")
    
    print()
    
    # ========================================
    # 3. Extract Stress-Strain Curves
    # ========================================
    print("3. Extracting stress-strain curves...")
    print("-" * 60)
    
    try:
        curves_xx = extract_stress_strain_curves(OUTPUT_DIR, 'xx')
        curves_yy = extract_stress_strain_curves(OUTPUT_DIR, 'yy')
        curves_zz = extract_stress_strain_curves(OUTPUT_DIR, 'zz')
        
        print(f"[OK] Extracted stress-strain curves for xx, yy, zz components")
        print(f"\nxx component:")
        print(f"  Max strain: {curves_xx['strain'].max():.4f}")
        print(f"  Max stress: {curves_xx['stress'].max():.2f} GPa")
        print(f"  Final stress: {curves_xx['stress'][-1]:.2f} GPa")
        
        print(f"\nyy component (should be ~0 for uniaxial):")
        print(f"  Final stress: {curves_yy['stress'][-1]:.4f} GPa")
        
        print(f"\nzz component (should be ~0 for uniaxial):")
        print(f"  Final stress: {curves_zz['stress'][-1]:.4f} GPa")
        
    except Exception as e:
        print(f"[ERROR] Error extracting stress-strain curves: {e}")
    
    print()
    
    # ========================================
    # 4. Plastic Strain Analysis (if checkpoint available)
    # ========================================
    if CHECKPOINT_FILE and os.path.exists(CHECKPOINT_FILE):
        print("4. Analyzing plastic strain evolution...")
        print("-" * 60)
        
        try:
            plastic_stats = compute_plastic_strain_evolution(CHECKPOINT_FILE)
            
            print(f"Mean von Mises plastic strain: {plastic_stats['eps_plastic_mean']:.6f}")
            print(f"Max von Mises plastic strain: {plastic_stats['eps_plastic_max']:.6f}")
            print(f"Active voxels (plastic strain > 0): {plastic_stats['active_voxels']}")
            
            total_voxels = np.prod(plastic_stats['eps_plastic_vm'].shape)
            active_fraction = plastic_stats['active_voxels'] / total_voxels * 100
            print(f"Active fraction: {active_fraction:.2f}%")
            
        except Exception as e:
            print(f"[ERROR] Error analyzing plastic strain: {e}")
        
        print()
    else:
        print("4. Plastic strain analysis skipped (no checkpoint file)")
        print("-" * 60)
        print("Set CHECKPOINT_FILE to analyze plastic strain evolution")
        print()
    
    # ========================================
    # 5. Generate Summary Report
    # ========================================
    print("5. Generating summary report...")
    print("-" * 60)
    
    report = generate_summary_report(OUTPUT_DIR, CHECKPOINT_FILE)
    print(report)
    
    # Save report to file
    report_file = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")
    print()
    
    # ========================================
    # 6. Create Plots
    # ========================================
    print("6. Creating visualization plots...")
    print("-" * 60)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Stress-strain curve
        ax = axes[0, 0]
        ax.plot(curves_xx['strain'] * 100, curves_xx['stress'], 'b-', linewidth=2)
        ax.set_xlabel('Strain ε_xx (%)')
        ax.set_ylabel('Stress σ_xx (GPa)')
        ax.set_title('Stress-Strain Curve')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cascade size distribution
        ax = axes[0, 1]
        if len(stats['cascade_sizes']) > 0:
            bin_edges, counts = stats['cascade_size_distribution']
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, counts, width=np.diff(bin_edges), 
                   edgecolor='black', alpha=0.7)
            ax.set_xlabel('Cascade Size (flips)')
            ax.set_ylabel('Count')
            ax.set_title('Cascade Size Distribution')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No cascade data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Plot 3: Flips per step
        ax = axes[1, 0]
        if len(stats['total_flips_per_step']) > 0:
            steps = np.arange(len(stats['total_flips_per_step']))
            ax.plot(steps, stats['total_flips_per_step'], 'r-', linewidth=1.5)
            ax.set_xlabel('Global Step')
            ax.set_ylabel('Total Flips')
            ax.set_title('Flips per Loading Step')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No flip data', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Plot 4: Transverse stresses (should be ~0 for uniaxial)
        ax = axes[1, 1]
        ax.plot(curves_xx['strain'] * 100, curves_yy['stress'], 
               'g-', label='σ_yy', linewidth=1.5)
        ax.plot(curves_xx['strain'] * 100, curves_zz['stress'], 
               'orange', label='σ_zz', linewidth=1.5, linestyle='--')
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
        ax.set_xlabel('Strain ε_xx (%)')
        ax.set_ylabel('Stress (GPa)')
        ax.set_title('Transverse Stresses (Uniaxial Test)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(OUTPUT_DIR, "postprocess_analysis.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[OK] Plots saved to: {plot_file}")
        
    except ImportError:
        print("[WARN] Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"[ERROR] Error creating plots: {e}")
    
    print()
    print("=" * 60)
    print("POST-PROCESSING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
