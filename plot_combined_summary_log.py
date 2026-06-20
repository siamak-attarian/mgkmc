import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_summary_log(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File '{filepath}' does not exist.")
        return None
        
    print(f"Parsing '{filepath}'...")
    
    headers = []
    data_rows = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('---'):
                    continue
                
                # Split line by whitespace
                parts = line.split()
                
                # Identify the header line (should contain 'Timestamp' and 'Step')
                if 'Timestamp' in parts and 'Step' in parts:
                    headers = parts
                    continue
                
                # Check if this is a data row (usually starts with date/time, e.g., '2026-06-19')
                # Timestamp column takes 2 parts: Date (parts[0]) and Time (parts[1])
                if headers and len(parts) >= len(headers) + 1:
                    # Merge the date and time to align with 'Timestamp' header single token
                    merged_parts = [parts[0] + ' ' + parts[1]] + parts[2:]
                    data_rows.append(merged_parts)
    except Exception as e:
        print(f"Warning: Error reading '{filepath}': {e}")
        return None
                
    if not headers:
        print(f"Warning: Could not find valid header containing 'Timestamp' and 'Step' in '{filepath}'.")
        return None
        
    if not data_rows:
        print(f"Warning: No data rows found in '{filepath}'.")
        return None
        
    # Find the indices of the strain and stress columns
    strain_col_idx = -1
    stress_col_idx = -1
    strain_col_name = ""
    stress_col_name = ""
    
    for i, h in enumerate(headers):
        if h.startswith('Eps_'):
            strain_col_idx = i
            strain_col_name = h
        elif h.startswith('Sig_'):
            stress_col_idx = i
            stress_col_name = h
            
    if strain_col_idx == -1 or stress_col_idx == -1:
        print(f"Warning: Could not automatically detect strain ('Eps_') and stress ('Sig_') columns in '{filepath}'.")
        return None
        
    # Extract numerical data
    strains = []
    stresses = []
    
    for row in data_rows:
        try:
            strains.append(float(row[strain_col_idx]))
            stresses.append(float(row[stress_col_idx]))
        except ValueError:
            # Skip rows that don't have valid numeric values
            continue
            
    if not strains:
        print(f"Warning: No valid numeric data extracted from '{filepath}'.")
        return None
            
    return np.array(strains), np.array(stresses), strain_col_name, stress_col_name

def main():
    parser = argparse.ArgumentParser(description="Plot combined summary log files.")
    parser.add_argument("--all", action="store_true", help="Search recursively in all subfolders. Default: only look into the first level of directories.")
    args = parser.parse_args()

    # Search for summary log files starting from the current directory
    current_dir = os.getcwd()
    print(f"Searching for 'summary_log' files in '{current_dir}'...")
    
    summary_files = []
    for root, dirs, files in os.walk(current_dir):
        # Calculate depth from current_dir
        rel_path = os.path.relpath(root, current_dir)
        if rel_path == ".":
            depth = 0
        else:
            depth = len(rel_path.split(os.sep))
            
        if not args.all and depth >= 1:
            dirs.clear()  # prevent recursing deeper than the first level subdirectories
            
        for file in files:
            # Look for summary_log.txt or files starting with summary_log
            if file.startswith("summary_log") and file.endswith(".txt"):
                filepath = os.path.join(root, file)
                summary_files.append(filepath)
            elif file == "summary_log":
                filepath = os.path.join(root, file)
                summary_files.append(filepath)
                
    if not summary_files:
        print("No summary log files found in the current directory or its subdirectories.")
        sys.exit(1)
        
    print(f"Found {len(summary_files)} summary log file(s):")
    for f in summary_files:
        print(f"  - {os.path.relpath(f, current_dir)}")
        
    # Set premium plotting styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    # We will use a colormap to automatically cycle colors nicely
    cmap = plt.get_cmap('tab10')
    
    plot_count = 0
    common_strain_label = "Strain"
    common_stress_label = "Stress"
    
    for idx, filepath in enumerate(summary_files):
        res = parse_summary_log(filepath)
        if res is None:
            continue
            
        strains, stresses, strain_label, stress_label = res
        common_strain_label = strain_label
        common_stress_label = stress_label
        
        # Determine a label based on the folder path
        rel_path = os.path.relpath(filepath, current_dir)
        folder_label = os.path.dirname(rel_path)
        if not folder_label or folder_label == ".":
            folder_label = os.path.basename(filepath)
        else:
            # Replace backslashes with slashes for clean layout
            folder_label = folder_label.replace('\\', '/')
            
        color = cmap(plot_count % 10)
        ax.plot(strains, stresses, linewidth=2.0, label=folder_label, color=color)
        plot_count += 1
        
    if plot_count == 0:
        print("Error: Could not parse any of the found summary log files.")
        sys.exit(1)
        
    # Clean design aesthetics
    ax.set_title("Combined Stress vs. Strain Curves", fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
    ax.set_xlabel(common_strain_label, fontsize=12, fontweight='bold', labelpad=10, color='#2C3E50')
    ax.set_ylabel(common_stress_label, fontsize=12, fontweight='bold', labelpad=10, color='#2C3E50')
    
    ax.tick_params(colors='#7F8C8D', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.6, color='#BDC3C7')
    
    # Place legend with a subtle background
    ax.legend(frameon=True, facecolor='white', edgecolor='#BDC3C7', framealpha=0.9, loc='best')
    
    plt.tight_layout()
    
    output_filename = "combined_stress_strain.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to '{os.path.join(current_dir, output_filename)}'")
    
    plt.show()

if __name__ == "__main__":
    main()
