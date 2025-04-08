import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .pose_encoder import MLPPoseTower, GCNPower, STGCNPower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_pose_tower(pose_tower_cfg, **kwargs):
    pose_tower = getattr(pose_tower_cfg, 'pose_tower', None)
    print('pose_tower_cfg', pose_tower_cfg)
    if pose_tower == 'MLP':
        return MLPPoseTower(hidden_dim = pose_tower_cfg.hidden_dim, num_joints = pose_tower_cfg.num_joints, max_frames = pose_tower_cfg.max_frames, use_joint_type = pose_tower_cfg.use_joint_type)
    elif pose_tower == 'GCN':
        return GCNPower(hidden_dim = pose_tower_cfg.hidden_dim, 
                        A = pose_tower_cfg.A, 
                        max_frames = pose_tower_cfg.max_frames, 
                        num_joints = pose_tower_cfg.num_joints, 
                        using_gat = pose_tower_cfg.using_gat)
    elif pose_tower == 'STGCN':
        return STGCNPower(hidden_dim = pose_tower_cfg.hidden_dim, 
                          A = pose_tower_cfg.A, 
                          max_frames = pose_tower_cfg.max_frames, 
                          num_joints = pose_tower_cfg.num_joints, 
                          using_gat = pose_tower_cfg.using_gat,
                          richer_frequency_representation = pose_tower_cfg.richer_frequency_representation)
    raise ValueError(f'Unknown pose tower: {pose_tower}')
