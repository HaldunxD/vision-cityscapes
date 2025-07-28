import argparse, torch, cv2, numpy as np
from pathlib import Path
from model import get_model
from utils import colorize_mask

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="checkpoints/segformer_e29.pt")
    p.add_argument("--input",   required=True, help="img file | dir | webcam id")
    p.add_argument("--outdir",  default="runs/")
    return p.parse_args()

def load_model(weights):
    ckpt = torch.load(weights, map_location="cpu")
    model = get_model(); model.load_state_dict(ckpt); model.eval().cuda()
    return model

@torch.no_grad()
def infer(model, img):
    import torchvision.transforms.functional as F
    x = F.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
    pred = model(pixel_values=x).logits.argmax(1)[0].cpu().numpy()
    return pred

def main():
    cfg=parse(); Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
    model = load_model(cfg.weights)
    if cfg.input.isdigit():
        cap = cv2.VideoCapture(int(cfg.input))
        while cap.isOpened():
            ret, frame = cap.read();  seg = infer(model, frame)
            vis = colorize_mask(seg, frame)
            cv2.imshow("Seg",vis); 0xFF & cv2.waitKey(1)
    else:
        p = Path(cfg.input)
        imgs = [p] if p.is_file() else sorted(p.glob("*.png"))
        for imfile in imgs:
            img = cv2.imread(str(imfile))
            seg = infer(model,img)
            vis = colorize_mask(seg,img)
            out = Path(cfg.outdir)/imfile.name
            cv2.imwrite(str(out),vis)
            print("saved",out)

if __name__=="__main__":
    main()
