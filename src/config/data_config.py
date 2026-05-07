# Configuration
RAW_DATA_PATH = "../data/raw"
OUTPUT_DIR = "../data/split"
LABEL_MAPPING_PATH = "../data/species_to_family.json"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Create output directories
import os

for split in ["train", "val", "test"]:
    os.makedirs(f"{OUTPUT_DIR}/{split}", exist_ok=True)
