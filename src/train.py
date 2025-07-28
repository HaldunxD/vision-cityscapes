import argparse, torch, os, wandb
from torch.utils.data import DataLoader
from accelerate import Accelerator
from dataset import CityscapesDataset
from model import get_model
from utils import colorize_mask, save_checkpoint

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="datasets/cityscapes")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    return ap.parse_args()

def main():
    cfg = parse()
    accelerator = Accelerator(mixed_precision="fp16")
    id2label = {i:c for i,c in enumerate([
        "road","sidewalk","building","wall","fence","pole","traffic light",
        "traffic sign","vegetation","terrain","sky","person","rider",
        "car","truck","bus","train","motorcycle","bicycle"])}
    label2id = {v:k for k,v in id2label.items()}

    train_ds = CityscapesDataset(cfg.data_root,"train")
    val_ds   = CityscapesDataset(cfg.data_root,"val")
    train_dl = DataLoader(train_ds, cfg.bs, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   cfg.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(19, id2label, label2id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    for epoch in range(cfg.epochs):
        model.train()
        for img, mask in train_dl:
            out = model(pixel_values=img, labels=mask)
            accelerator.backward(out.loss)
            optimizer.step(); optimizer.zero_grad()
        # --- quick val ---
        model.eval(); miou=0; cnt=0
        with torch.no_grad():
            for img, mask in val_dl:
                logits = model(pixel_values=img).logits
                preds = logits.argmax(1)
                miou += (preds==mask).float().mean().item(); cnt+=1
        miou /= cnt
        accelerator.print(f"{epoch=:02d}  mIoU={miou:.4f}")
        save_checkpoint(model, accelerator, f"checkpoints/segformer_e{epoch}.pt")
        if wandb.run: wandb.log({"mIoU":miou,"epoch":epoch})

if __name__ == "__main__":
    main()
