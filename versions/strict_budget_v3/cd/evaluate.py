from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from .data import rgb_tensor
from .metrics import Metrics


@torch.no_grad()
def predict(model, a, b, device, tile=256, batch_size=4):
    if tile < 16 or tile % 16 or batch_size < 1:
        raise ValueError('tile需为16的正整数倍，batch_size为正。')
    width, height = a.size
    probs = np.zeros((height, width), np.float32)
    tensors_a, tensors_b, positions = [], [], []
    def flush():
        if not positions:return
        probability=model(torch.stack(tensors_a).to(device),torch.stack(tensors_b).to(device)).sigmoid()[:,0].cpu().numpy()
        for i,(x,y,w,h) in enumerate(positions):probs[y:y+h,x:x+w]=probability[i,:h,:w]
        tensors_a.clear();tensors_b.clear();positions.clear()
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            w, h = min(tile, width-x), min(tile, height-y)
            for image, buffer in ((a,tensors_a),(b,tensors_b)):
                t=rgb_tensor(image.crop((x,y,x+w,y+h)))
                buffer.append(F.pad(t,(0,tile-w,0,tile-h),mode='replicate'))
            positions.append((x,y,w,h))
            if len(positions)==batch_size:flush()
    flush()
    return probs


def evaluate(model, records, device, tile=256, threshold=0.5, tolerance=2, visual_dir=None, max_visual=8, brightness=1.0):
    model.eval()
    metric = Metrics(tolerance)
    per_image = []
    for i, r in enumerate(records):
        with Image.open(r['a']) as im:
            a = im.convert('RGB')
        with Image.open(r['b']) as im:
            b = im.convert('RGB')
        if brightness != 1:
            b = ImageEnhance.Brightness(b).enhance(brightness)
        with Image.open(r['label']) as im:
            truth = np.asarray(im.convert('L')) > 0
        pred = predict(model, a, b, device, tile) >= threshold
        metric.update(pred, truth)
        local = Metrics(tolerance)
        local.update(pred, truth)
        per_image.append(dict(name=r['name'], **local.result()))
        if visual_dir and i < max_visual:
            out = Path(visual_dir)
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred.astype(np.uint8) * 255).save(out / (r['name'] + '_prediction.png'))
            error = np.zeros((*truth.shape, 3), np.uint8)
            error[pred & truth] = (255, 255, 255)
            error[pred & ~truth] = (255, 0, 0)
            error[~pred & truth] = (0, 100, 255)
            panels = [a, b, Image.fromarray(truth.astype(np.uint8)*255).convert('RGB'), Image.fromarray(pred.astype(np.uint8)*255).convert('RGB'), Image.fromarray(error)]
            scale = min(1, 256 / a.width)
            panels = [p.resize((int(a.width*scale), int(a.height*scale))) for p in panels]
            canvas = Image.new('RGB', (panels[0].width * 5, panels[0].height))
            for j, p in enumerate(panels):
                canvas.paste(p, (j*p.width, 0))
            canvas.save(out / (r['name'] + '_panels.png'))
    result = metric.result()
    result.update({'images': len(records), 'threshold': threshold, 'boundary_tolerance_px': tolerance,
                   'brightness_factor_t2': brightness, 'per_image': per_image})
    return result
