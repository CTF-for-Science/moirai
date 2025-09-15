import torch
import pickle
import numpy as np
from pathlib import Path
from ctf4science.eval_module import evaluate

# file dir
file_dir = Path(__file__).parent.parent
# pickle dir
pickle_dir = file_dir / 'pickles'

def main() -> None:
    """
    Main function to evaluate scores for Moirai.
    """
    # Get names of all pickles available
    pickle_names = [p.name for p in pickle_dir.glob('*.pkl')]

    for pickle_name in pickle_names:
        print(f"> {pickle_name}")
        # Load pickle
        with open(pickle_dir / pickle_name, 'rb') as f:
            pred_data = pickle.load(f)
        dataset_name = pickle_name.split('_')[0]
        pair_id = int(pickle_name.split('_')[1].split('.')[0])

        print(f"  pred_data shape: {pred_data.shape}")

        # Evaluate predictions using default metrics
        results = evaluate(dataset_name, pair_id, pred_data)

        print(f"  {results}")

if __name__ == '__main__':
    main()