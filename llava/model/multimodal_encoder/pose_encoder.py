import torch
import torch.nn as nn
import re

import torch.nn.functional as F

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
    def __init__(self, hidden_dim: int, num_joints: int, num_frames: int, use_joint_type=False):
        super().__init__()
        self.input_dim = 6  # (x, y, xmin, ymin, xmax, ymax)
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.use_joint_type = use_joint_type

        # Pose projector (MLP)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, hidden_dim)
        )

        # Learnable mask token for invisible joints
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.mask_token)

        # Positional encodings
        self.joint_index_embed = nn.Parameter(torch.randn(1, 1, num_joints, hidden_dim))  # (1, 1, J, H)
        self.frame_index_embed = nn.Parameter(torch.randn(1, num_frames, 1, hidden_dim))  # (1, F, 1, H)

        if use_joint_type:
            # Optional joint type embedding (e.g., hand vs foot)
            self.joint_type_embed = nn.Embedding(num_joints, hidden_dim)

    def forward(self, pose_with_bbox: torch.Tensor, visibility: torch.Tensor):
        """
        pose_with_bbox: (B, F, J, 6) - (x, y, xmin, ymin, xmax, ymax)
        visibility:     (B, F, J)
        """
        B, F, J, _ = pose_with_bbox.shape
        assert J == self.num_joints and F <= self.num_frames

        # Project pose to hidden_dim
        pose_embed = self.net(pose_with_bbox)  # (B, F, J, hidden_dim)

        # Base joint and frame position embeddings
        joint_pos = self.joint_index_embed[:, :, :J, :]           # (1, 1, J, H)
        frame_pos = self.frame_index_embed[:, :F, :, :]          # (1, F, 1, H)
        pose_embed = pose_embed + joint_pos + frame_pos

        if self.use_joint_type:
            # Treat joint index as type for embedding
            joint_ids = torch.arange(J, device=pose_embed.device).view(1, 1, J).expand(B, F, J)
            joint_type_emb = self.joint_type_embed(joint_ids)    # (B, F, J, H)
            pose_embed = pose_embed + joint_type_emb

        # Apply visibility masking
        mask = self.mask_token.expand(B, F, J, -1)               # (B, F, J, H)
        pose_embed = torch.where(visibility.unsqueeze(-1).bool(), pose_embed, mask)

        pose_embed = pose_embed.reshape(B, F * J, -1)
        return pose_embed  # (B, F, J, hidden_dim)



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


class PoseGCNProjector(nn.Module):
    def __init__(self, hidden_dim, A, using_gat=False):
        super().__init__()
        self.A = A

        self.linear = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 32))
        if using_gat:
            self.gcn1 = GATLayer(32, 128)
            self.gcn2 = GATLayer(128, 512)
            self.gcn3 = GATLayer(512, 2048)
        else:
            self.gcn1 = GCNLayer(32, 128)
            self.gcn2 = GCNLayer(128, 512)
            self.gcn3 = GCNLayer(512, 2048)

        self.output_linear = nn.Linear(2048, hidden_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 32))
        nn.init.xavier_uniform_(self.mask_token)

    def forward(self, pose_with_bbox, visibility):
        B, F, J, _ = pose_with_bbox.shape
        # v = visibility[:, t].unsqueeze(-1)
        # x = x * v
        x = self.linear(pose_with_bbox)
        mask = self.mask_token.expand(B, J, F, -1)
        x = torch.where(visibility.unsqueeze(-1).bool(), x, mask)
        x = F.gelu(x)
        x = self.gcn1(x, self.A)
        x = F.gelu(x)
        x = self.gcn2(x, self.A)
        x = F.gelu(x)
        x = self.gcn3(x, self.A)
        x = F.gelu(x)
        x = self.output_linear(x)
        B, F, J, _ = x.shape
        x = x.reshape(B, F * J, -1)
        return x


class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, using_gat=False, richer_frequency_representation=False):
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
            self.spatial_gcn = GCNLayer(in_channels_for_spatial_gcn, out_channels_for_spatial_gcn)
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
        x = F.gelu(x)
        x = self.temporal_conv(x)
        x = self.bn(x)

        x = x + res
        x = x.permute(0, 2, 3, 1).contiguous() # x (B, F, J, C)
        return x


class PoseSTGCNProjector(nn.Module):
    def __init__(self, hidden_dim, A, using_gat=False, richer_frequency_representation=False):
        super().__init__()
        self.A = A
        self.linear = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 32))
        self.stgcn1 = STGCNLayer(32, 128, using_gat=using_gat, richer_frequency_representation=richer_frequency_representation)
        self.stgcn2 = STGCNLayer(128, 512, using_gat=using_gat, richer_frequency_representation=richer_frequency_representation)
        self.stgcn3 = STGCNLayer(512, 2048, using_gat=using_gat, richer_frequency_representation=richer_frequency_representation)
        self.output_linear = nn.Linear(2048, hidden_dim)


        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 32))
        nn.init.xavier_uniform_(self.mask_token)

    def forward(self, pose_with_bbox, visibility):
        B, F, J, C = pose_with_bbox.shape
        # x = pose_with_bbox * visibility.unsqueeze(-1)
        x = self.linear(pose_with_bbox)
        mask = self.mask_token.expand(B, J, F, -1)
        x = torch.where(visibility.unsqueeze(-1).bool(), x, mask)
        x = F.gelu(x)
        x = self.stgcn1(x, self.A)
        x = F.gelu(x)
        x = self.stgcn2(x, self.A)
        x = F.gelu(x)
        x = self.stgcn3(x, self.A)
        x = F.gelu(x)
        x = self.output_linear(x)
        B, F, J, C = x.shape
        x = x.reshape(B, F * J, C)
        return x
