import numpy as np, cv2, torch, json, os
CITY_COLORS = np.array([
 [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156],
 [190,153,153], [153,153,153], [250,170, 30], [220,220,  0],
 [107,142, 35], [152,251,152], [ 70,130,180], [220, 20, 60],
 [255,  0,  0], [  0,  0,142], [  0,  0, 70], [  0, 60,100],
 [  0, 80,100], [  0,  0,230], [119, 11, 32]], dtype=np.uint8)

def colorize_mask(mask, img=None):
    color_mask = CITY_COLORS[mask]
    if img is None: return color_mask
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    blend = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
    return blend

def save_checkpoint(model, accelerator, path):
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model).state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(unwrapped, path)
