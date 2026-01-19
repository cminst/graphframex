# Define the path to the data, model, logs, results, and colors
#
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODEL_DIR = os.path.join(REPO_ROOT, "model")
LOG_DIR = os.path.join(REPO_ROOT, "logs")
RESULT_DIR = os.path.join(REPO_ROOT, "results")
MASK_DIR = os.path.join(REPO_ROOT, "mask")
