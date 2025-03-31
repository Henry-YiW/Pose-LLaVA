import os
import numpy as np
import json
import argparse
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from pose_encoder import PoseEncoder
from pose_llava_model import PoseLlavaModel


def load_model(model_path, device="cuda"):
    """Load the Pose-LLaVA model"""
    # Load the base LLM
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, base_model, _, _ = load_pretrained_model(
        model_path, 
        None, 
        model_name,
        device_map=device
    )
    
    # Create and load pose encoder
    pose_encoder = PoseEncoder(base_model.config)
    pose_encoder.to(device)
    
    # Create and load PoseLlavaModel
    model = PoseLlavaModel(
        base_model,
        pose_encoder,
        projector_type="mlp2x_gelu"  # Use the same type as during training
    )
    
    # Load projector weights if they exist separately
    projector_path = os.path.join(model_path, "mm_projector.bin")
    if os.path.exists(projector_path):
        model.mm_projector.load_state_dict(torch.load(projector_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def load_validation_data(val_data_path):
    """Load the validation data"""
    try:
        if val_data_path.endswith('.npy'):
            data = np.load(val_data_path, allow_pickle=True).item()
            return data['samples']
        elif val_data_path.endswith('.json'):
            with open(val_data_path, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise ValueError(f"Unsupported file format: {val_data_path}")
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return None


def load_action_mapping(mapping_path):
    """Load action ID to text mapping"""
    try:
        with open(mapping_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading action mapping: {e}")
        return {}


def extract_action_from_response(response, action_mapping):
    """Extract the action from the model's response"""
    # Try to match the standard format "The action of the pose sequence is X."
    pattern = r"action of the pose sequence is (.+?)[\.|\n]"
    match = re.search(pattern, response, re.IGNORECASE)
    
    if match:
        predicted_text = match.group(1).strip().lower()
        
        # Check if the text matches any of the action names
        for action_id, action_name in action_mapping.items():
            if action_name.lower() == predicted_text:
                return int(action_id)
        
        # If no exact match is found, try to find the closest one
        max_similarity = 0
        best_action_id = None
        
        for action_id, action_name in action_mapping.items():
            # Simple string similarity measure
            similarity = sum(c1 == c2 for c1, c2 in zip(action_name.lower(), predicted_text))
            if similarity > max_similarity:
                max_similarity = similarity
                best_action_id = int(action_id)
        
        if best_action_id is not None:
            return best_action_id
    
    # Default case: couldn't extract action
    return None


def evaluate_model(model, tokenizer, validation_data, action_mapping, conv_mode="vicuna_v1", device="cuda"):
    """Evaluate the Pose-LLaVA model on validation data"""
    from llava.conversation import conv_templates
    
    true_labels = []
    predicted_labels = []
    results = []
    
    for sample in tqdm(validation_data, desc="Evaluating"):
        # Extract data
        if 'action' in sample:
            true_action = sample['action']
            if isinstance(true_action, np.ndarray):
                true_action = true_action.item()
        else:
            continue  # Skip samples without action labels
        
        x_coords = torch.tensor(sample['x'], dtype=torch.float32).unsqueeze(0).to(device)
        y_coords = torch.tensor(sample['y'], dtype=torch.float32).unsqueeze(0).to(device)
        visibility = torch.tensor(sample['visibility'], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Prepare prompt
        prompt = "Recognize the action from this pose sequence."
        
        # Set up conversation
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                x_coords=x_coords,
                y_coords=y_coords,
                visibility=visibility,
                do_sample=False,
                temperature=0.1,
                max_new_tokens=50,
                use_cache=True
            )
        
        # Decode and extract response
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_text.split(conv.roles[1] + ":")[-1].strip()
        
        # Extract predicted action
        predicted_action = extract_action_from_response(response, action_mapping)
        
        # Store results
        true_labels.append(true_action)
        predicted_labels.append(predicted_action if predicted_action is not None else -1)
        
        results.append({
            "sample_id": sample.get('file_name', '').replace('.mat', ''),
            "true_action": true_action,
            "true_action_text": action_mapping.get(str(true_action), f"action_{true_action}"),
            "predicted_action": predicted_action,
            "predicted_action_text": action_mapping.get(str(predicted_action), f"action_{predicted_action}") if predicted_action is not None else "unknown",
            "response": response,
            "correct": predicted_action == true_action
        })
    
    # Calculate metrics
    valid_predictions = [p for p, t in zip(predicted_labels, true_labels) if p != -1]
    valid_true_labels = [t for p, t in zip(predicted_labels, true_labels) if p != -1]
    
    if len(valid_predictions) > 0:
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        report = classification_report(valid_true_labels, valid_predictions, output_dict=True)
        conf_matrix = confusion_matrix(valid_true_labels, valid_predictions)
    else:
        accuracy = 0
        report = {}
        conf_matrix = np.array([])
    
    # Calculate overall stats
    total_samples = len(true_labels)
    correct_samples = sum(1 for r in results if r["correct"])
    unknown_samples = sum(1 for p in predicted_labels if p == -1)
    
    metrics = {
        "accuracy": accuracy,
        "total_samples": total_samples,
        "correct_samples": correct_samples,
        "unknown_samples": unknown_samples,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    return results, metrics


def main(model_path, val_data_path, mapping_path, output_dir, conv_mode="vicuna_v1", device="cuda"):
    """Run evaluation on the Pose-LLaVA model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path, device)
    
    # Load action mapping
    print(f"Loading action mapping from {mapping_path}...")
    action_mapping = load_action_mapping(mapping_path)
    
    # Load validation data
    print(f"Loading validation data from {val_data_path}...")
    val_data = load_validation_data(val_data_path)
    if not val_data:
        print("Failed to load validation data")
        return
    
    # Evaluate model
    print("Evaluating model...")
    results, metrics = evaluate_model(model, tokenizer, val_data, action_mapping, conv_mode, device)
    
    # Save results
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\n===== Evaluation Summary =====")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct predictions: {metrics['correct_samples']} ({metrics['correct_samples'] / metrics['total_samples'] * 100:.2f}%)")
    print(f"Unknown predictions: {metrics['unknown_samples']} ({metrics['unknown_samples'] / metrics['total_samples'] * 100:.2f}%)")
    print(f"Accuracy (excluding unknown): {metrics['accuracy'] * 100:.2f}%")
    print(f"Results saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pose-LLaVA model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Pose-LLaVA model")
    parser.add_argument("--val-data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--mapping", type=str, required=True, help="Path to action mapping JSON file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for evaluation results")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1", help="Conversation mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    main(args.model_path, args.val_data, args.mapping, args.output_dir, args.conv_mode, args.device)