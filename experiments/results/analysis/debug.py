import os
import glob

def list_all_files():
    """List all experiment files to debug the issue"""
    # Check Dian's directory
    dian_dir = "../../from_dian"
    print(f"Checking directory: {dian_dir}")
    if os.path.exists(dian_dir):
        print("Directory exists!")
        files = os.listdir(dian_dir)
        for file in files:
            if file.endswith(".json") and not file.startswith("."):
                print(f"Found file: {file}")
                # Try to directly open this file
                try:
                    with open(os.path.join(dian_dir, file), 'r') as f:
                        content = f.read(100)  # Read just the first 100 chars
                        print(f"  File starts with: {content[:50]}...")
                except Exception as e:
                    print(f"  Error reading file: {e}")
    else:
        print(f"Directory {dian_dir} does not exist!")
        
    # Try other directory paths
    print("\nTrying absolute path:")
    abs_dir = os.path.abspath("../../from_dian")
    print(f"Absolute path: {abs_dir}")
    
    print("\nListing all json files using glob:")
    for file in glob.glob("../../from_dian/*.json"):
        print(f"Glob found: {file}")
    
    print("\nChecking from project root:")
    root_dir = "../../../experiments/results/from_dian"
    if os.path.exists(root_dir):
        print(f"Directory {root_dir} exists!")
        files = os.listdir(root_dir)
        for file in files:
            if file.endswith(".json"):
                print(f"Found file from root: {file}")
    
    print("\nTrying with glob from root:")
    for file in glob.glob("../../../experiments/results/from_dian/*.json"):
        print(f"Glob from root found: {file}")

if __name__ == "__main__":
    list_all_files() 