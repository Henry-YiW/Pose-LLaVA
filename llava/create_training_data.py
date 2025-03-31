import os
import numpy as np
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import Dataset, DatasetDict

def load_pose_data(npy_path):
    """Load pose data from NPY file"""
    try:
        data = np.load(npy_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading {npy_path}: {e}")
        return None


def load_action_mapping(mapping_path):
    """Load action mapping from JSON file"""
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading action mapping from {mapping_path}: {e}")
        print("Using default action mapping")
        # Default mapping if file not found
        return {str(i): f"action_{i}" for i in range(20)}


def create_conversation_data(pose_data, action_mapping):
    """Create conversation data for Pose-LLaVA training"""
    conversation_data = []
    
    for sample in tqdm(pose_data, desc="Creating conversation data"):
        action_id = sample['action']
        # Convert to string key for JSON lookup
        action_text = action_mapping.get(str(action_id), f"action_{action_id}")
        
        # Create conversation entry
        entry = {
            "id": sample.get('file_name', '').replace('.mat', ''),
            "action": int(action_id),
            "action_text": action_text,
            "x": sample['x'].tolist(),
            "y": sample['y'].tolist(),
            "visibility": sample['visibility'].tolist(),
            "bbox": sample['bbox'].tolist(),
            'nframes': sample['nframes'],
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
        conversation_data.append(entry)
    


    dataset = Dataset.from_list(conversation_data)
    # dataset.save_to_disk(output_path)
    # # Save to JSON file
    # with open(output_path, 'w') as f:
    #     json.dump(conversation_data, f, indent=2)
    
    # print(f"Created conversation data with {len(conversation_data)} samples, saved to {output_path}")
    return dataset


def main(train_data_path, mapping_path, output_dir, val_data_path=None):
    """Main function to create training data for Pose-LLaVA"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load action mapping
    action_mapping = load_action_mapping(mapping_path)
    print(action_mapping)
    # Load training data
    train_data = load_pose_data(train_data_path)
    print(len(train_data))
    if train_data is None or len(train_data) == 0:
        print("Failed to load training data")
        return
    
    # Create training conversation data
    # train_output_path = os.path.join(output_dir, 'pose_train')
    train_conversations = create_conversation_data(train_data, action_mapping)
    
    # Load and process validation data if provided
    if val_data_path:
        val_data = load_pose_data(val_data_path)
        if val_data is not None and len(val_data) > 0:
            # val_output_path = os.path.join(output_dir, 'pose_val')
            val_conversations = create_conversation_data(val_data, action_mapping)
    

    # Create a DatasetDict to hold both splits
    dataset_dict = DatasetDict({
        "train": train_conversations,
        "validation": val_conversations
    })
    dataset_dict.save_to_disk(output_dir)
    # Create a README file with dataset information
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# Pose-LLaVA Training Dataset\n\n")
        f.write(f"This dataset contains {len(train_conversations)} training samples.\n\n")
        
        if val_data_path:
            f.write(f"It also includes a validation set with {len(val_conversations)} samples.\n\n")
        
        f.write("## Data Format\n\n")
        f.write("Each sample contains:\n")
        f.write("- `id`: Unique identifier\n")
        f.write("- `action`: Numerical action label\n")
        f.write("- `action_text`: Human-readable action label\n")
        f.write("- `x`: X-coordinates of pose keypoints (76 frames, 13 keypoints)\n")
        f.write("- `y`: Y-coordinates of pose keypoints (76 frames, 13 keypoints)\n")
        f.write("- `visibility`: Visibility flags for keypoints\n")
        f.write("- `conversations`: Conversation format for training\n\n")
        
        f.write("## Actions\n\n")
        f.write("The dataset contains the following actions:\n\n")
        for action_id, action_name in action_mapping.items():
            f.write(f"- {action_id}: {action_name}\n")
    
    print(f"Created README file at {readme_path}")
    print(f"Dataset preparation completed. Files saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data for Pose-LLaVA")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data NPY file")
    parser.add_argument("--val-data", type=str, default=None, help="Path to validation data NPY file")
    parser.add_argument("--mapping", type=str, required=True, help="Path to action mapping JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for training data")
    
    args = parser.parse_args()
    
    main(args.train_data, args.mapping, args.output_dir, args.val_data)