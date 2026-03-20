import importlib
import logging
import os
import sys

CUSTOM_MODULES_DIR_NAME = "custom_modules"

# import any custom modules
current_working_dir = os.getcwd()
custom_modules_path = os.path.join(current_working_dir, CUSTOM_MODULES_DIR_NAME)
if os.path.exists(custom_modules_path) and os.path.isdir(custom_modules_path):
    # Add 'custom_modules' folder to sys.path
    sys.path.append(os.getcwd())
    logging.getLogger("terratorch").info(f"Found {CUSTOM_MODULES_DIR_NAME}")
    importlib.import_module(CUSTOM_MODULES_DIR_NAME)
