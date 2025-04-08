import math
import torch
import torch.nn as nn
import re

import torch.nn.functional as TorchF

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)



class MLPPoseTower(nn.Module):
    def __init__(self, hidden_dim: int, num_joints: int, max_frames: int, use_joint_type=False):
        super().__init__()
        self.input_dim = 6  # (x, y, xmin, ymin, xmax, ymax)
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.max_frames = max_frames
        self.use_joint_type = use_joint_type

        self.intermediate_dim = 54

        # Pose projector (MLP)
        self.joint_net = nn.Sequential(
            nn.Linear(self.input_dim, 18),
            nn.GELU(),
            nn.Linear(18, self.intermediate_dim)
        )

        # self.frame_net = nn.Sequential(
        #     nn.Linear(self.input_dim, 24),
        #     nn.GELU(),
        #     nn.Linear(24, 96),
        #     nn.GELU(),
        #     nn.Linear(96, 384)
        # )

        # self.frame_net = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(96 * num_joints, nhead=8),
        #     num_layers=3
        # )

        self.frame_net = nn.Sequential(
            nn.Linear(self.intermediate_dim * num_joints, 1404),
            nn.GELU(),
            nn.Linear(1404, 2808),
            nn.GELU(),
            nn.Linear(2808, hidden_dim)
        )

        # Learnable mask token for invisible joints
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.intermediate_dim))
        nn.init.xavier_uniform_(self.mask_token)

        # Positional encodings
        # non-learnable emdding would provide generalization to unseen joints
        # self.joint_index_embed = nn.Parameter(torch.randn(1, 1, num_joints, hidden_dim))  # (1, 1, J, H)
        # self.frame_index_embed = nn.Parameter(torch.randn(1, max_frames, 1, hidden_dim))  # (1, F, 1, H)

        self.register_buffer("frame_index_embed", self.build_sinusoidal_embedding(max_frames, hidden_dim))
        self.register_buffer("joint_index_embed", self.build_sinusoidal_embedding(num_joints, hidden_dim))


        if use_joint_type:
            # Optional joint type embedding (e.g., hand vs foot)
            self.joint_type_embed = nn.Embedding(num_joints, hidden_dim)

    def build_sinusoidal_embedding(self, num_positions: int, dim: int):
        position = torch.arange(num_positions).unsqueeze(1)  # (num_positions, 1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.view(1, num_positions, dim)  # shape: (1, F, D)
    
    def forward(self, pose_with_bbox: torch.Tensor, visibility: torch.Tensor):
        """
        pose_with_bbox: (B, F, J, 6) - (x, y, xmin, ymin, xmax, ymax)
        visibility:     (B, F, J)
        """
        # print('pose_with_bbox', pose_with_bbox)
        # print('pose_with_bbox length', len(pose_with_bbox))
        B, F, J, _ = pose_with_bbox.shape
        try:
            assert J == self.num_joints and F <= self.max_frames
        except Exception as e:
            print('Exepected Max Frames:', self.max_frames, 'Actual Max Frames:', F)
            raise e
        pose_with_bbox = pose_with_bbox.to(dtype=self.joint_net[0].weight.dtype)
        # Project pose to hidden_dim
        pose_embed = self.joint_net(pose_with_bbox)  # (B, F, J, hidden_dim)

        if self.use_joint_type:
            # Treat joint index as type for embedding
            joint_ids = torch.arange(J, device=pose_embed.device).view(1, 1, J).expand(B, F, J)
            joint_type_emb = self.joint_type_embed(joint_ids)    # (B, F, J, H)
            pose_embed = pose_embed + joint_type_emb

        # Apply visibility masking
        mask = self.mask_token.expand(B, F, J, -1)               # (B, F, J, H)
        pose_embed = torch.where(visibility.unsqueeze(-1).bool(), pose_embed, mask)

        # Check if the pose_embed is right

        pose_embed = pose_embed.reshape(B, F, -1)

        pose_embed = self.frame_net(pose_embed)

        # Base frame position embeddings
        frame_pos = self.frame_index_embed[:, :F, :]          # (1, F, H)
        pose_embed = pose_embed + frame_pos

        return pose_embed  # (B, F, hidden_dim)



class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.A_weight = nn.Parameter(torch.ones(num_joints, num_joints))  # same shape (J, J)



    def forward(self, x, A, learnable_weight=False):
        B, F, J, C = x.shape
        x = x.permute(0, 1, 3, 2)  # (B, F, C, J) for matmul

        if learnable_weight:
            A_eff = self.A_weight * A  # (J, J)
        else:
            A_eff = A

        A_eff = A_eff.to(x.device)
        x = torch.matmul(x, A_eff)  # (B, F, C, J)
        x = x.permute(0, 1, 3, 2)   # (B, F, J, C)
        x = self.linear(x)          # linear over last dim
        return x

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.alpha = 1.0
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        # self.attn = nn.Parameter(torch.Tensor(out_channels * 2))
        # nn.init.xavier_uniform_(self.attn.view(2, -1))
        self.attn = nn.Linear(out_channels * 2, 1, bias=False)

    def forward(self, x, A):
        B, F, J,  C = x.shape
        h = self.linear(x)  # (B, F, J, out_channels)

        scores = []
        for i in range(J):
            for j in range(J):
                if A[i, j] == 0:
                    continue
                hi = h[:, :, i]  # (B, F, C)
                hj = h[:, :, j]  # (B, F, C)
                a_input = torch.cat([hi, hj], dim=-1)  # (B, F, 2C)
                e_ij = F.leaky_relu(self.attn(a_input), negative_slope=0.2)  # (B, F, 1)
                scores.append(((i, j), e_ij))

        # Initialize attention matrix
        attn_matrix = torch.zeros(B, F, J, J, device=x.device)
        for (i, j), score in scores:
            attn_matrix[:, :, i, j] = score

        attn_matrix = attn_matrix + A.unsqueeze(0) * self.alpha
        attn_matrix = attn_matrix.masked_fill(A.unsqueeze(0) == 0, float('-inf'))
        attn_matrix = F.softmax(attn_matrix, dim=-1)

        # attn_matrix = F.softmax(attn_matrix, dim=-1)  # normalize over neighbors
        # attn_matrix = attn_matrix * self.A.unsqueeze(0)  # broadcast A to (B, J, J)
        out = torch.bmm(attn_matrix.reshape(B * F, J, J), h.reshape(B * F, J, C))  # (B, F, J, C)
        out = out.reshape(B, F, J, C)
        return out


class GCNPower(nn.Module):
    def __init__(self, hidden_dim, A, max_frames, num_joints, using_gat=False):
        super().__init__()
        self.A = A
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.max_frames = max_frames
        self.linear = nn.Sequential(nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 24))
        if using_gat:
            self.gcn1 = GATLayer(24, 48)
            self.gcn2 = GATLayer(48, 96)
            self.gcn3 = GATLayer(96, 192)
        else:
            self.gcn1 = GCNLayer(24, 48, num_joints)
            self.gcn2 = GCNLayer(48, 96, num_joints)
            self.gcn3 = GCNLayer(96, 192, num_joints)

        self.frame_net = nn.Sequential(
            nn.Linear(192 * num_joints, 2496 * 2),
            nn.GELU(),
            nn.Linear(2496 * 2, 2496),
            nn.GELU(),
            nn.Linear(2496, hidden_dim)
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 24))
        nn.init.xavier_uniform_(self.mask_token)


        # Positional encodings
        # non-learnable emdding would provide generalization to unseen joints
        # self.joint_index_embed = nn.Parameter(torch.randn(1, 1, num_joints, hidden_dim))  # (1, 1, J, H)
        # self.frame_index_embed = nn.Parameter(torch.randn(1, max_frames, 1, hidden_dim))  # (1, F, 1, H)

        self.register_buffer("frame_index_embed", self.build_sinusoidal_embedding(max_frames, hidden_dim))
        self.register_buffer("joint_index_embed", self.build_sinusoidal_embedding(num_joints, hidden_dim))

    def build_sinusoidal_embedding(self, num_positions: int, dim: int):
        position = torch.arange(num_positions).unsqueeze(1)  # (num_positions, 1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.view(1, num_positions, dim)  # shape: (1, F, D)
    
    def forward(self, pose_with_bbox, visibility):
        pose_with_bbox = pose_with_bbox.to(dtype=self.linear[0].weight.dtype)
        B, F, J, _ = pose_with_bbox.shape
        # v = visibility[:, t].unsqueeze(-1)
        # x = x * v
        x = self.linear(pose_with_bbox)
        # print('x shape', x.shape)
        # print('visibility shape', visibility.shape)
        # print('mask shape', self.mask_token.shape)
        mask = self.mask_token.expand(B, F, J, -1)
        x = torch.where(visibility.unsqueeze(-1).bool(), x, mask)
        x = TorchF.gelu(x)
        x = self.gcn1(x, self.A)
        x = TorchF.gelu(x)
        x = self.gcn2(x, self.A)
        x = TorchF.gelu(x)
        x = self.gcn3(x, self.A)
        x = TorchF.gelu(x)
        B, F, J, _ = x.shape
        x = x.reshape(B, F, -1)
        x = self.frame_net(x)
        x = x + self.frame_index_embed[:, :F, :]
        return x


class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints, kernel_size=3, using_gat=False, richer_frequency_representation=False):
        super().__init__()
        in_channels_for_spatial_gcn = in_channels
        out_channels_for_spatial_gcn = out_channels
        in_channels_for_temporal_conv = out_channels
        out_channels_for_temporal_conv = out_channels

        if richer_frequency_representation:
            in_channels_for_spatial_gcn = in_channels
            out_channels_for_spatial_gcn = in_channels
            in_channels_for_temporal_conv = in_channels
            out_channels_for_temporal_conv = out_channels

        if using_gat:
            self.spatial_gcn = GATLayer(in_channels_for_spatial_gcn, out_channels_for_spatial_gcn)
        else:
            self.spatial_gcn = GCNLayer(in_channels_for_spatial_gcn, out_channels_for_spatial_gcn, num_joints)
        self.temporal_conv = nn.Conv2d(in_channels_for_temporal_conv, out_channels_for_temporal_conv, (kernel_size, 1), padding=(kernel_size // 2, 0))
        self.bn = nn.BatchNorm2d(out_channels_for_temporal_conv)
        self.using_gat = using_gat

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):
        # x (B, F, J, C)
        res = x.permute(0, 3, 1, 2).contiguous()  # (B, C, F, J)
        res = self.residual(res)

        x = self.spatial_gcn(x, A) # x (B, F, J, C)
        x = x.permute(0, 3, 1, 2).contiguous() # x (B, C, F, J)
        x = TorchF.gelu(x)
        x = self.temporal_conv(x)
        x = self.bn(x)

        x = x + res
        x = x.permute(0, 2, 3, 1).contiguous() # x (B, F, J, C)
        return x


class STGCNPower(nn.Module):
    def __init__(self, hidden_dim, A, max_frames, num_joints, using_gat=False, richer_frequency_representation=False):
        super().__init__()
        self.A = A
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.max_frames = max_frames
        self.linear = nn.Sequential(nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 24))
        self.stgcn1 = STGCNLayer(24, 48, num_joints, kernel_size=3, using_gat=using_gat, richer_frequency_representation=richer_frequency_representation)
        self.stgcn2 = STGCNLayer(48, 96, num_joints, kernel_size=3, using_gat=using_gat, richer_frequency_representation=richer_frequency_representation)
        self.stgcn3 = STGCNLayer(96, 192, num_joints, kernel_size=3, using_gat=using_gat, richer_frequency_representation=richer_frequency_representation)
        
        self.frame_net = nn.Sequential(
            nn.Linear(192 * num_joints, 2496 * 2),
            nn.GELU(),
            nn.Linear(2496 * 2, 2496),
            nn.GELU(),
            nn.Linear(2496, hidden_dim)
        )


        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 24))
        nn.init.xavier_uniform_(self.mask_token)


        # Positional encodings
        # non-learnable emdding would provide generalization to unseen joints
        # self.joint_index_embed = nn.Parameter(torch.randn(1, 1, num_joints, hidden_dim))  # (1, 1, J, H)
        # self.frame_index_embed = nn.Parameter(torch.randn(1, max_frames, 1, hidden_dim))  # (1, F, 1, H)

        self.register_buffer("frame_index_embed", self.build_sinusoidal_embedding(max_frames, hidden_dim))
        self.register_buffer("joint_index_embed", self.build_sinusoidal_embedding(num_joints, hidden_dim))

    def build_sinusoidal_embedding(self, num_positions: int, dim: int):
        position = torch.arange(num_positions).unsqueeze(1)  # (num_positions, 1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(num_positions, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.view(1, num_positions, dim)  # shape: (1, F, D)
    
    def forward(self, pose_with_bbox, visibility):
        pose_with_bbox = pose_with_bbox.to(dtype=self.linear[0].weight.dtype)
        B, F, J, C = pose_with_bbox.shape
        # x = pose_with_bbox * visibility.unsqueeze(-1)
        x = self.linear(pose_with_bbox)
        # print('x shape', x.shape)
        # print('visibility shape', visibility.shape)
        # print('mask shape', self.mask_token.shape)
        mask = self.mask_token.expand(B, F, J, -1)
        x = torch.where(visibility.unsqueeze(-1).bool(), x, mask)
        x = TorchF.gelu(x)
        x = self.stgcn1(x, self.A)
        x = TorchF.gelu(x)
        x = self.stgcn2(x, self.A)
        x = TorchF.gelu(x)
        x = self.stgcn3(x, self.A)
        x = TorchF.gelu(x)
        B, F, J, C = x.shape
        x = x.reshape(B, F, -1)
        x = self.frame_net(x)
        x = x + self.frame_index_embed[:, :F, :]
        return x
