import importlib
import json
import os
from dogms.src.utils import load_config

# --- USER SETTINGS ---
SCRIPT_NAME = "test"        # Name of the script in src/ without .py
CONFIG_FILE = "config_test.json" # Name of config file in configs/

#Note: you can run the main with python -m dogms.main
# ----------------------

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "configs", CONFIG_FILE)
    config = load_config(config_path)

    # Dynamically import the module
    try:
        module = importlib.import_module(f"dogms.src.{SCRIPT_NAME}")
    except ModuleNotFoundError:
        raise ImportError(f"Script '{SCRIPT_NAME}' not found in src/")

    # Check and run 'run(config)' function
    if hasattr(module, "run"):
        module.run(config)
    else:
        raise AttributeError(f"The module '{SCRIPT_NAME}' does not have a 'run(config)' function.")

if __name__ == "__main__":
    main()
