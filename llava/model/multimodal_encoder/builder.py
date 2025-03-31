import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .pose_encoder import MLPPoseTower

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
    if pose_tower == 'MLP':
        return MLPPoseTower(pose_tower_cfg)
    # elif pose_tower == 'GCN':
    #     return GCNPoseTower(pose_tower_cfg)
    raise ValueError(f'Unknown pose tower: {pose_tower}')
