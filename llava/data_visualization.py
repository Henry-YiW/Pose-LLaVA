import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from matplotlib.patches import Rectangle
import scipy.io
import json


def load_mat_file(file_path):
    """Load a single .mat file"""
    try:
        mat_data = scipy.io.loadmat(file_path)
        return mat_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def plot_keypoints_frame(ax, x, y, visibility=None, bbox=None, connections=None, action_name=None):
    """Plot keypoints for a single frame"""
    # Clear previous frame
    ax.clear()
    
    # If visibility is not provided, assume all keypoints are visible
    if visibility is None:
        visibility = np.ones_like(x)
    
    # Plot visible keypoints
    visible_idx = np.where(visibility > 0.5)[0]
    ax.scatter(x[visible_idx], y[visible_idx], c='blue', s=20)
    
    # Plot less visible keypoints
    less_visible_idx = np.where(visibility <= 0.5)[0]
    if len(less_visible_idx) > 0:
        ax.scatter(x[less_visible_idx], y[less_visible_idx], c='gray', s=10, alpha=0.5)
    
    # Draw connections between keypoints if provided
    if connections:
        for connection in connections:
            i, j = connection
            if i < len(x) and j < len(x) and visibility[i] > 0.5 and visibility[j] > 0.5:
                ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=0.7)
    
    # Draw bounding box if provided
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    # Set title
    if action_name:
        ax.set_title(f"Action: {action_name}")
    
    # Set axis limits with some padding
    all_x = x[visible_idx]
    all_y = y[visible_idx]
    if len(all_x) > 0 and len(all_y) > 0:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        padding = max(max_x - min_x, max_y - min_y) * 0.1
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
    
    # Invert y-axis since image coordinates have origin at top-left
    ax.invert_yaxis()
    
    # Remove ticks for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])


def create_animation(mat_file_path, output_path=None, connections=None, action_mapping=None):
    """Create animation from .mat file"""
    # Load .mat file
    data = load_mat_file(mat_file_path)
    if not data:
        return None
    
    # Extract data
    x = data['x']  # (80, 13)
    y = data['y']  # (80, 13)
    visibility = data.get('visibility', np.ones_like(x))  # (80, 13)
    bbox = data.get('bbox', None)  # (80, 4) or None
    action_id = data.get('action', None)
    
    # Default human pose connections (simplified skeleton)
    if connections is None:
        # These connections are just an example and should be adapted to your keypoint structure
        connections = [
            (0, 1), (1, 2), (2, 3),  # Head to left shoulder to left elbow to left wrist
            (0, 4), (4, 5), (5, 6),  # Head to right shoulder to right elbow to right wrist
            (0, 7),                   # Head to hip center
            (7, 8), (8, 9), (9, 10),  # Hip center to left hip to left knee to left ankle
            (7, 11), (11, 12), (12, 13)  # Hip center to right hip to right knee to right ankle
        ]
    
    # Get action name
    action_name = None
    if action_id is not None:
        action_id = action_id.item() if hasattr(action_id, 'item') else action_id
        if action_mapping and action_id in action_mapping:
            action_name = action_mapping[action_id]
        else:
            action_name = f"Action {action_id}"
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Function to update the plot for each frame
    def update(frame):
        if frame < len(x):
            current_bbox = None
            if bbox is not None and frame < len(bbox):
                current_bbox = bbox[frame]
            plot_keypoints_frame(
                ax, 
                x[frame], 
                y[frame], 
                visibility[frame] if frame < len(visibility) else None,
                current_bbox,
                connections,
                action_name
            )
        return ax,
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(x), interval=100, blit=False)
    
    # Save animation if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ani.save(output_path, writer='pillow', fps=10)
        plt.close()
        return output_path
    else:
        plt.tight_layout()
        plt.show()
        return ani


def visualize_multiple_samples(data_dir, output_dir, num_samples=5, action_mapping=None):
    """Visualize multiple pose samples from the dataset"""
    # Load action mapping if provided as a path
    if action_mapping and isinstance(action_mapping, str) and os.path.exists(action_mapping):
        with open(action_mapping, 'r') as f:
            action_mapping = json.load(f)
    
    # Get all .mat files
    mat_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mat')]
    
    # Select a subset of files if there are too many
    if len(mat_files) > num_samples:
        np.random.seed(42)  # For reproducibility
        selected_files = np.random.choice(mat_files, num_samples, replace=False)
    else:
        selected_files = mat_files
    
    # Create animations for each selected file
    for i, file_path in enumerate(selected_files):
        file_name = os.path.basename(file_path).replace('.mat', '')
        output_path = os.path.join(output_dir, f"{file_name}.gif")
        print(f"Creating animation for {file_name} ({i+1}/{len(selected_files)})...")
        create_animation(file_path, output_path, action_mapping=action_mapping)
    
    print(f"Created {len(selected_files)} animations in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pose data")
    parser.add_argument("--input", type=str, required=True, help="Path to .mat file or directory containing .mat files")
    parser.add_argument("--output", type=str, default=None, help="Path to save visualization/animations")
    parser.add_argument("--mapping", type=str, default=None, help="Path to action mapping JSON file")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize if input is a directory")
    
    args = parser.parse_args()
    
    # Load action mapping if provided
    action_mapping = None
    if args.mapping:
        try:
            with open(args.mapping, 'r') as f:
                action_mapping = json.load(f)
        except Exception as e:
            print(f"Error loading action mapping: {e}")
    
    if os.path.isdir(args.input):
        # Visualize multiple samples
        output_dir = args.output if args.output else "pose_visualizations"
        visualize_multiple_samples(args.input, output_dir, args.num_samples, action_mapping)
    else:
        # Visualize a single sample
        output_path = args.output if args.output else None
        create_animation(args.input, output_path, action_mapping=action_mapping)