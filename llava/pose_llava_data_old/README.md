# Pose-LLaVA Training Dataset

This dataset contains 1860 training samples.

It also includes a validation set with 466 samples.

## Data Format

Each sample contains:
- `id`: Unique identifier
- `action`: Numerical action label
- `action_text`: Human-readable action label
- `x`: X-coordinates of pose keypoints (76 frames, 13 keypoints)
- `y`: Y-coordinates of pose keypoints (76 frames, 13 keypoints)
- `visibility`: Visibility flags for keypoints
- `conversations`: Conversation format for training

## Actions

The dataset contains the following actions:

- 0: baseball_pitch
- 1: baseball_swing
- 2: bench_press
- 3: bowl
- 4: clean_and_jerk
- 5: golf_swing
- 6: jump_rope
- 7: jumping_jacks
- 8: pullup
- 9: pushup
- 10: situp
- 11: squat
- 12: strum_guitar
- 13: tennis_forehand
- 14: tennis_serve
