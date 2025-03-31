import torch
from typing import Dict, List, Sequence
from dataclasses import dataclass
from llava.constants import IGNORE_INDEX

@dataclass
class PoseDataCollator:
    """
    Data collator for pose sequence data that pads sequences to the maximum length in the batch.
    """
    
    tokenizer: any  # Tokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        print('instances shape', instances[0])
        input_ids, labels = tuple([instance[key] for instance in instances] 
                                 for key in ("input_ids", "labels"))
        
        # Pad input_ids and labels to same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )
        
        # Truncate to maximum length supported by the model
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        # Batch pose keypoints data
        x_coords = torch.stack([instance["x_coords"] for instance in instances])
        y_coords = torch.stack([instance["y_coords"] for instance in instances])
        visibility = torch.stack([instance["visibility"] for instance in instances])
        action_labels = torch.stack([instance["action_label"] for instance in instances])
        
        # Create attention mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "visibility": visibility,
            "action_labels": action_labels,
        }
        
        return batch