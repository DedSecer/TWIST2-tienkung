import os
import yaml

def generate_config_yaml(folder_path, output_yaml_path):
    # Specify the root path
    config = {
        "root_path": folder_path,
        "motions": []
    }

    # Recursively list all .pkl files under folder_path.
    # `motions[].file` must be relative to `root_path` so MotionLib can resolve it.
    pkl_entries = []
    for current_root, dirs, files in os.walk(folder_path):
        # Make traversal deterministic
        dirs.sort()
        for file_name in sorted(files):
            if not file_name.endswith(".pkl"):
                continue
            abs_path = os.path.join(current_root, file_name)
            rel_path = os.path.relpath(abs_path, folder_path)
            pkl_entries.append(rel_path)

    for rel_path in sorted(pkl_entries):
        config["motions"].append({
            "file": rel_path,
            "weight": 1.0,
            "description": "general movement"
        })

    # Write the configuration to the YAML file
    with open(output_yaml_path, "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False)

# Specify your folder path and output YAML file
folder_path = "/home/dedsecer/Downloads/TWIST2_full"
output_yaml_path = "twist.yaml"

generate_config_yaml(folder_path, output_yaml_path)
print(f"YAML configuration file generated: {output_yaml_path}")