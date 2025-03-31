import torch
import torch.nn as nn

class PoseEncoder(nn.Module):
    """
    Encoder for pose keypoints sequences, replacing the vision tower in LLaVA.
    Takes keypoint coordinates and visibility as input and produces embeddings.
    """
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.hidden_size = config.hidden_size
        self.num_keypoints = 13
        self.sequence_length = 80
        self.keypoint_dim = 2  # x, y coordinates
        self.use_visibility = True
        self.input_dim = self.num_keypoints * (self.keypoint_dim + (1 if self.use_visibility else 0))
        
        # Layers
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Temporal transformer layer to capture sequence dynamics
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=self.hidden_size * 4,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x_coords, y_coords, visibility=None):
        """
        Process pose keypoint sequences.
        
        Args:
            x_coords: (batch_size, sequence_length, num_keypoints)
            y_coords: (batch_size, sequence_length, num_keypoints)
            visibility: (batch_size, sequence_length, num_keypoints)
            
        Returns:
            Tensor of shape (batch_size, sequence_length, hidden_size)
        """
        batch_size = x_coords.shape[0]
        
        # Concatenate coordinates and visibility
        if self.use_visibility and visibility is not None:
            # Reshape to (batch_size, sequence_length, num_keypoints * 3)
            pose_features = torch.stack([
                x_coords.view(batch_size, self.sequence_length, self.num_keypoints),
                y_coords.view(batch_size, self.sequence_length, self.num_keypoints),
                visibility.view(batch_size, self.sequence_length, self.num_keypoints)
            ], dim=-1)
        else:
            # Reshape to (batch_size, sequence_length, num_keypoints * 2)
            pose_features = torch.stack([
                x_coords.view(batch_size, self.sequence_length, self.num_keypoints),
                y_coords.view(batch_size, self.sequence_length, self.num_keypoints)
            ], dim=-1)
        
        # Flatten keypoints dimension
        pose_features = pose_features.reshape(batch_size, self.sequence_length, -1)
        
        # Project to embedding space
        embeddings = self.embedding_projection(pose_features)
        
        # Apply temporal transformer
        temporal_embeddings = self.temporal_transformer(embeddings)
        
        # Final projection
        output_embeddings = self.output_projection(temporal_embeddings)
        
        return output_embeddings
    
    @property
    def device(self):
        """Return the device this model is on"""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Return the dtype of this model"""
        return next(self.parameters()).dtype
    
    @property
    def config(self):
        """Mock config to be compatible with vision tower"""
        class Config:
            def __init__(self, hidden_size):
                self.hidden_size = hidden_size
                self.image_size = None
                self.patch_size = None
                
        return Config(self.hidden_size)
        
    def is_loaded(self):
        """Always return True to be compatible with vision tower"""
        return True