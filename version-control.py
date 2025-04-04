import subprocess

libraries = [
    "numpy", "pandas", "torch", "tqdm", "scikit-learn",
    "pandas", "matplotlib", "seaborn", "networkx", "torchtext",
    "optuna", "torch_geometric", "pm4py"
]

def get_library_versions(libs):
    for lib in sorted(set(libs)):  # Rimuove duplicati e ordina
        result = subprocess.run(["pip", "show", lib], capture_output=True, text=True)
        version = "Not Installed"
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                version = line.split(":")[1].strip()
                break
        print(f"{lib}: {version}")

if __name__ == "__main__":
    get_library_versions(libraries)
