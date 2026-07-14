import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_summary_log(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        sys.exit(1)
        
    print(f"Parsing '{filepath}'...")
    
    headers = []
    data_rows = []
    
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
                
    if not headers:
        print("Error: Could not find valid header containing 'Timestamp' and 'Step' in the log file.")
        sys.exit(1)
        
    if not data_rows:
        print("Error: No data rows found in the log file.")
        sys.exit(1)
        
    # Find the indices of the strain and stress columns
    strain_col_idx = -1
    stress_col_idx = -1
    strain_col_name = ""
    stress_col_name = ""
    
    for i, h in enumerate(headers):
        h_lower = h.lower()
        if h.startswith('Eps_') or h.startswith('Eng_strain_') or 'strain' in h_lower or 'eps' in h_lower:
            strain_col_idx = i
            strain_col_name = h
        elif h.startswith('Sig_') or 'sig' in h_lower or 'stress' in h_lower:
            stress_col_idx = i
            stress_col_name = h
            
    if strain_col_idx == -1 or stress_col_idx == -1:
        print("Error: Could not automatically detect strain and stress columns.")
        print(f"Found headers: {headers}")
        sys.exit(1)
        
    print(f"Detected columns: Strain='{strain_col_name}', Stress='{stress_col_name}'")
    
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
            
    return np.array(strains), np.array(stresses), strain_col_name, stress_col_name

def main():
    # Determine the log file path
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = 'summary_log.txt'
        
    strains, stresses, strain_label, stress_label = parse_summary_log(log_file)
    
    # Set premium plotting styles
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Plot curve with a sleek styling
    ax.plot(strains, stresses, color='#1A5F7A', linewidth=2.5, label='Simulation Response')
    
    # Clean design aesthetics
    ax.set_title("Stress vs. Strain Curve", fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
    ax.set_xlabel(strain_label, fontsize=12, fontweight='bold', labelpad=10, color='#2C3E50')
    ax.set_ylabel(stress_label, fontsize=12, fontweight='bold', labelpad=10, color='#2C3E50')
    
    ax.tick_params(colors='#7F8C8D', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.6, color='#BDC3C7')
    
    # Tight layout and show/save
    plt.tight_layout()
    
    output_filename = "stress_strain.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to '{output_filename}'")
    
    plt.show()

if __name__ == "__main__":
    main()
