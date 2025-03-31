import os
import numpy as np
import json
import scipy.io
import argparse
from tqdm import tqdm
from pathlib import Path


def process_mat_files(input_dir, output_path, split_ratio=0.8):
    """Process all .mat files in the input directory and save as JSON/NPY"""
    # Get a list of all .mat files in the input directory
    mat_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.mat')])
    
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return
    
    print(f"Found {len(mat_files)} .mat files in {input_dir}")
    
    # Get action mapping (we'll update this with string actions)
    print('Loading action mapping...')
    action_mapping = {}
    
    # Process each .mat file
    all_data = []
    action_to_id = {}  # Map string actions to numeric IDs
    next_action_id = max([int(aid) for aid in action_mapping.keys()], default=-1) + 1
    
    for file_path in tqdm(mat_files, desc="Processing MAT files"):
        data = scipy.io.loadmat(file_path)
        if data and data['action'] is not None:
            action = data['action'][0]
            # Handle string actions by assigning numeric IDs
            if isinstance(action, str):
                if action not in action_to_id:
                    action_to_id[action] = next_action_id
                    # Add to the action mapping
                    action_mapping[str(next_action_id)] = action
                    next_action_id += 1
                # Store the original string action and convert to numeric ID
                data['original_action'] = str(action)
                data['action'] = action_to_id[action]
            data['file_name'] = file_path.split('/')[-1]
            data['file_path'] = file_path
            all_data.append(data)
    
    # Shuffle the data
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(all_data)
    
    # Split into train and validation sets
    split_idx = int(len(all_data) * split_ratio)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Save train and validation data
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(output_path).split('.')[0]
    
    # Save as NPY (faster to load for training)
    train_npy_path = os.path.join(output_dir, f"{base_name}_train.npy")
    val_npy_path = os.path.join(output_dir, f"{base_name}_val.npy")
    
    # np.save(train_npy_path, convert_to_numpy_dict(train_data))
    # np.save(val_npy_path, convert_to_numpy_dict(val_data))
    np.save(train_npy_path, train_data)
    np.save(val_npy_path, val_data)
    
    print(f"Saved training data ({len(train_data)} samples) to {train_npy_path}")
    print(f"Saved validation data ({len(val_data)} samples) to {val_npy_path}")
    
    # Save updated action mapping with string actions included
    mapping_path = os.path.join(output_dir, f"{base_name}_action_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(action_mapping, f, indent=2)
    
    print(f"Saved action mapping to {mapping_path}")
    
    # Save data info (stats about the dataset)
    actions_count = {}
    for sample in all_data:
        action_id = sample['action']
        if action_id in actions_count:
            actions_count[action_id] += 1
        else:
            actions_count[action_id] = 1
    
    data_info = {
        'total_samples': len(all_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'num_actions': len(set(s['action'] for s in all_data if s['action'] is not None)),
        'actions_distribution': {action_mapping.get(str(k), f"action_{k}"): v for k, v in actions_count.items()},
        'num_frames': all_data[0]['x'].shape[0] if all_data else 0,
        'num_keypoints': all_data[0]['x'].shape[1] if all_data else 0,
    }
    
    info_path = os.path.join(output_dir, f"{base_name}_info.json")
    with open(info_path, 'w') as f:
        json.dump(data_info, f, indent=2)
    
    print(f"Saved dataset info to {info_path}")
    
    # Also create a sample file for Pose-LLaVA training format
    sample_data = []
    for i, data in enumerate(train_data[:5]):  # Just use the first 5 samples as examples
        action_id = data['action']
        action_text = action_mapping.get(str(action_id), f"action_{action_id}")
        print(data)
        sample = {
            "file_name": data['file_name'],
            "action": action_id,
            "action_text": action_text,
            "x": data['x'].tolist(),
            "y": data['y'].tolist(),
            "visibility": data['visibility'].tolist(),
            "conversations": [
                {
                    "from": "human",
                    "value": "Recognize the action from this pose sequence."
                },
                {
                    "from": "gpt",
                    "value": f"The action of the pose sequence is {action_text}."
                }
            ]
        }
        sample_data.append(sample)
    
    sample_path = os.path.join(output_dir, f"{base_name}_sample.json")
    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Saved sample data format to {sample_path}")
    
    return {
        "train_path": train_npy_path,
        "val_path": val_npy_path,
        "mapping_path": mapping_path,
        "info_path": info_path,
        "sample_path": sample_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .mat files and convert to NPY/JSON format")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .mat files")
    parser.add_argument("--output-path", type=str, required=True, help="Output path prefix for saved data")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/val split ratio (default: 0.8)")

    args = parser.parse_args()
    process_mat_files(args.input_dir, args.output_path, args.split_ratio)