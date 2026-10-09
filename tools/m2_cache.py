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
    if len(set(int(x) for x in artifact['pid_order'])) != len(artifact['pid_order']):
        raise ValueError('pid_order must contain unique training identities')
    # These fields make the train-only provenance auditable at load time.  A
    # relation matrix without them can be numerically valid while originating
    # from query/gallery data or a different CLIP preprocessing pipeline.
    meta = artifact.get('metadata', {})
    if meta.get('split') != 'train' or meta.get('modality') != 'RGB':
        raise ValueError('M2-4 cache must declare split=train and modality=RGB')
    if meta.get('feature_dim') != 512 or meta.get('normalize') != 'l2':
        raise ValueError('M2-4 cache metadata does not describe normalized 512-D CLIP centers')
    return True

def build_relation_cache(centers, pid_order, output, metadata=None):
    centers=torch.as_tensor(centers,dtype=torch.float32); centers=centers/(centers.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    metadata = dict(metadata or {})
    metadata.setdefault('split', 'train')
    metadata.setdefault('modality', 'RGB')
    metadata.setdefault('feature_dim', 512)
    metadata.setdefault('normalize', 'l2')
    artifact={'centers':centers.cpu(),'relation':(centers@centers.t()).cpu(),'pid_order':list(map(int,pid_order)),'source':'raw CLIP RGB train centers','version':1,'metadata':metadata}
    validate_artifact(artifact); os.makedirs(os.path.dirname(os.path.abspath(output)),exist_ok=True); torch.save(artifact,output); return artifact
