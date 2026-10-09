import hashlib, random, numpy as np, torch
def file_digest(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read_clip(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    return obj.state_dict() if isinstance(obj,torch.nn.Module) else obj
def provenance(cache,clip):
    return {'split':cache.get('split'),'clip_path':cache.get('clip_path',str(clip)),'clip_digest':file_digest(clip),'preprocess':cache.get('preprocess',{}),'pid_map':cache.get('pid_map',{})}
def rng_state():
    return {'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
