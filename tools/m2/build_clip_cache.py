"""Generate a train-only CLIP cache for M2-1 from WHU-MARS train images."""
import argparse, os, torch
from PIL import Image
from torchvision import transforms
from datasets.whu_mars import WHU_MARS
from model.backbones.vit_pytorch import vit_base_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True); ap.add_argument('--clip', required=True)
    ap.add_argument('--output', required=True); ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(); device = torch.device(a.device if a.device != 'cuda' or torch.cuda.is_available() else 'cpu')
    ds = WHU_MARS(root=a.data_root, modalities=['RGB','IR','Thermal'], verbose=False)
    model = vit_base_clip(img_size=(256,128), stride_size=[16,16], drop_path_rate=0.0).to(device).eval()
    model.load_param(a.clip)
    tfm = transforms.Compose([transforms.Resize((256,128)), transforms.ToTensor(), transforms.Normalize(
        [0.48145466,0.4578275,0.40821073], [0.26862954,0.26130258,0.27577711])])
    rows=[]
    with torch.no_grad():
        for modality, items in ds.train.items():
            for path,pid,cam,_ in items:
                rows.append((path,int(pid),int(cam),modality))
        feats=[]
        for start in range(0,len(rows),a.batch_size):
            batch=torch.stack([tfm(Image.open(r[0]).convert('RGB')) for r in rows[start:start+a.batch_size]]).to(device)
            feats.append(torch.nn.functional.normalize(model(batch).float() @ model.clip_proj.float(), dim=1).cpu())
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    pids=torch.tensor([r[1] for r in rows]); cams=torch.tensor([r[2] for r in rows]); mods=torch.tensor([['RGB','IR','Thermal'].index(r[3]) for r in rows]); platforms=torch.isin(cams,torch.tensor([5,6])).long()
    torch.save({'feature':torch.cat(feats), 'features':torch.cat(feats), 'pid':pids, 'pids':pids, 'cams':cams, 'modalities':mods, 'modality':mods, 'platform':platforms, 'pid_map':{int(i):int(i) for i in torch.unique(pids).tolist()}, 'split':'train', 'clip_path':a.clip, 'preprocess':{'size':[256,128], 'normalize':'CLIP'}}, a.output)
    print('saved train-only cache:', a.output, 'rows=', len(rows))

if __name__ == '__main__': main()
