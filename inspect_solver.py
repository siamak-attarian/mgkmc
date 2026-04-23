content = open('mgkmc/solver.py', 'rb').read().decode('utf-8')

# Extract spectral_solver_2d body
start = content.find('def spectral_solver_2d')
end   = content.find('\ndef run_mixed_simulation_2d')
print(content[start:end])
