import os

def rename():
    path = r'd:\GoogleDrive\2-MGKMC\mgkmc'
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.py') or f.endswith('.md'):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                    if 'ThermalSimulation' in content:
                        content = content.replace('ThermalSimulation', 'ThermalSimulation')
                        with open(fpath, 'w', encoding='utf-8', errors='ignore') as file:
                            file.write(content)
                        print(f'Updated {fpath}')
                except Exception as e:
                    print(f"Error on {fpath}: {e}")

if __name__ == "__main__":
    rename()
