"""M2-4 cache validation and deterministic relation artifact builder."""
import os, torch

def validate_artifact(artifact, cfg=None):
    if not isinstance(artifact, dict) or 'relation' not in artifact:
        raise ValueError('M2-4 relation cache must be a dict containing relation')
    rel=artifact['relation']
    if not torch.is_tensor(rel) or rel.ndim != 2 or rel.shape[0] != rel.shape[1]:
        raise ValueError('relation cache must be square [num_train_ids,num_train_ids]')
    if not torch.isfinite(rel).all(): raise ValueError('relation cache contains non-finite values')
    if 'centers' not in artifact or artifact['centers'].ndim != 2 or artifact['centers'].shape[1] != 512:
        raise ValueError('M2-4 cache must include RGB CLIP centers [ids,512]')
    if artifact['centers'].shape[0] != rel.shape[0]: raise ValueError('center/relation identity count mismatch')
    if 'pid_order' not in artifact or len(artifact['pid_order']) != rel.shape[0]: raise ValueError('missing pid_order mapping')
    return True

def build_relation_cache(centers, pid_order, output):
    centers=torch.as_tensor(centers,dtype=torch.float32); centers=centers/(centers.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    artifact={'centers':centers.cpu(),'relation':(centers@centers.t()).cpu(),'pid_order':list(map(int,pid_order)),'source':'raw CLIP RGB train centers','version':1}
    validate_artifact(artifact); os.makedirs(os.path.dirname(os.path.abspath(output)),exist_ok=True); torch.save(artifact,output); return artifact
