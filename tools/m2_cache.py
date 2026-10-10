"""M2-4 centers/relation provenance; never invent a source declaration."""
import os
import torch
from utils.m4_artifacts import validate_centers, file_hash

def validate_artifact(artifact, cfg, num_classes=None):
    count = validate_centers(artifact, cfg, num_classes)
    relation = artifact['relation'].float()
    if relation.shape != (count, count) or not torch.isfinite(relation).all():
        raise ValueError('invalid teacher cosine relation')
    centers = artifact['centers'].float()
    torch.testing.assert_close(relation, centers @ centers.t(), rtol=1e-5, atol=1e-6)
    return True

def build_relation_cache(center_artifact, output, cfg):
    validate_centers(center_artifact, cfg)
    artifact = dict(center_artifact)
    centers = center_artifact['centers'].detach().float().cpu()
    artifact['relation'] = centers @ centers.t()
    artifact['centers_sha256'] = file_hash(cfg.M2.RGB_CENTERS)
    validate_artifact(artifact, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    torch.save(artifact, output)
    return artifact

def load_relation_bank(cfg, num_classes, device='cpu'):
    artifact = torch.load(cfg.M2.RELATION_BANK, map_location='cpu', weights_only=False)
    validate_artifact(artifact, cfg, num_classes)
    if artifact['centers_sha256'] != file_hash(cfg.M2.RGB_CENTERS):
        raise ValueError('M2-4 center file and relation bank are from different preparations')
    centers = torch.load(cfg.M2.RGB_CENTERS, map_location='cpu', weights_only=False)
    validate_centers(centers, cfg, num_classes)
    torch.testing.assert_close(artifact['centers'], centers['centers'], rtol=0, atol=0)
    artifact['relation'] = artifact['relation'].detach().float().to(device)
    return artifact
