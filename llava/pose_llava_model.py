import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

class PoseLlavaModel(nn.Module):
    """
    LLaVA model with a pose encoder instead of a vision encoder.
    Adapts the LlavaLlamaForCausalLM class to work with pose keypoint sequences.
    """
    
    def __init__(self, base_llm, pose_encoder, projector_type="mlp"):
        super().__init__()
        self.base_model = base_llm
        self.pose_encoder = pose_encoder
        self.config = base_llm.config
        
        # Create projector from pose embeddings to language embedding space
        hidden_size = base_llm.config.hidden_size
        pose_hidden_size = pose_encoder.hidden_size
        
        if projector_type == "linear":
            self.mm_projector = nn.Linear(pose_hidden_size, hidden_size)
        elif projector_type == "mlp":
            self.mm_projector = nn.Sequential(
                nn.Linear(pose_hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        elif projector_type == "mlp2x_gelu":
            self.mm_projector = nn.Sequential(
                nn.Linear(pose_hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
            
        # Initialize projector
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def get_pose_embeddings(self, x_coords, y_coords, visibility=None):
        """Get embeddings from pose keypoints"""
        # Forward pass through pose encoder
        pose_embeddings = self.pose_encoder(x_coords, y_coords, visibility)
        
        # Project to language model embedding space
        projected_embeddings = self.mm_projector(pose_embeddings)
        
        return projected_embeddings

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        x_coords: Optional[torch.FloatTensor] = None,
        y_coords: Optional[torch.FloatTensor] = None,
        visibility: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with pose data integration
        """
        if inputs_embeds is None and x_coords is not None and y_coords is not None:
            # Process pose sequences and get embeddings
            pose_embeds = self.get_pose_embeddings(x_coords, y_coords, visibility)
            
            # Get text embeddings
            if input_ids is not None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
            
            # Create combined embeddings
            batch_size = pose_embeds.shape[0]
            pose_tokens_per_seq = pose_embeds.shape[1]
            
            # Calculate combined sequence length
            if inputs_embeds is not None:
                seq_length = inputs_embeds.shape[1] + pose_tokens_per_seq
                
                # Create a new inputs_embeds tensor that includes pose embeddings
                new_inputs_embeds = torch.zeros(
                    (batch_size, seq_length, inputs_embeds.shape[2]),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device
                )
                
                # First, add pose embeddings
                new_inputs_embeds[:, :pose_tokens_per_seq] = pose_embeds
                
                # Then, add text embeddings
                new_inputs_embeds[:, pose_tokens_per_seq:] = inputs_embeds
                
                inputs_embeds = new_inputs_embeds
                
                # Update attention mask to include pose tokens
                if attention_mask is not None:
                    pose_attention_mask = torch.ones(
                        (batch_size, pose_tokens_per_seq),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([pose_attention_mask, attention_mask], dim=1)
                
                # Update position_ids if provided
                if position_ids is not None:
                    pose_position_ids = torch.arange(
                        pose_tokens_per_seq,
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    ).expand(batch_size, -1)
                    text_position_ids = position_ids + pose_tokens_per_seq
                    position_ids = torch.cat([pose_position_ids, text_position_ids], dim=1)
            else:
                # Only pose embeddings
                inputs_embeds = pose_embeds
        
        # Forward pass through base model
        outputs = self.base_model(
            input_ids=None,  # We're using inputs_embeds instead
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        print('outputs', outputs)
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        x_coords: Optional[torch.FloatTensor] = None,
        y_coords: Optional[torch.FloatTensor] = None,
        visibility: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text with pose sequence input
        """
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
            
        if x_coords is not None and y_coords is not None:
            # Process pose data and create embeddings
            pose_embeds = self.get_pose_embeddings(x_coords, y_coords, visibility)
            
            if input_ids is not None:
                # Get text embeddings for prompts
                text_embeds = self.base_model.model.embed_tokens(input_ids)
                
                # Calculate combined sequence length
                batch_size = pose_embeds.shape[0]
                pose_tokens_per_seq = pose_embeds.shape[1]
                
                # Create combined embeddings
                seq_length = text_embeds.shape[1] + pose_tokens_per_seq
                
                inputs_embeds = torch.zeros(
                    (batch_size, seq_length, text_embeds.shape[2]),
                    dtype=text_embeds.dtype,
                    device=text_embeds.device
                )
                
                # First, add pose embeddings
                inputs_embeds[:, :pose_tokens_per_seq] = pose_embeds
                
                # Then, add text embeddings
                inputs_embeds[:, pose_tokens_per_seq:] = text_embeds
                
                # Update attention mask to include pose tokens
                if attention_mask is not None:
                    pose_attention_mask = torch.ones(
                        (batch_size, pose_tokens_per_seq),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([pose_attention_mask, attention_mask], dim=1)
                
                # Update position_ids if provided
                if position_ids is not None:
                    pose_position_ids = torch.arange(
                        pose_tokens_per_seq,
                        dtype=position_ids.dtype,
                        device=position_ids.device
                    ).expand(batch_size, -1)
                    text_position_ids = position_ids + pose_tokens_per_seq
                    position_ids = torch.cat([pose_position_ids, text_position_ids], dim=1)
            else:
                # Only pose embeddings
                inputs_embeds = pose_embeds
        else:
            # No pose data, just use text embeddings
            inputs_embeds = self.base_model.model.embed_tokens(input_ids)
        
        # Generate text outputs
        return self.base_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )