import cv2
from ultralytics import YOLO
import torch
from PIL import Image

model = YOLO('yolov8m-seg.pt')

def make_mask(input_image_path, id):
    img = cv2.imread(input_image_path)
    results = model.predict(source=img, save=False, save_txt=False, stream=True,imgsz=[1280,720])
    all_masks_combined = None

    for result in results:
        masks = result.masks.data
        if all_masks_combined is None:
            all_masks_combined = masks.any(dim=0)
        else:
            all_masks_combined = torch.logical_or(all_masks_combined, masks.any(dim=0))

    all_masks_combined = (all_masks_combined.int() * 255).cpu().numpy()

    cv2.imwrite(f'./mask/mask_{id}.png', all_masks_combined)
    img = Image.open(f'./mask/mask_{id}.png')
    img_resized = img.resize((1280, 720))
    img_resized.save(f'./mask/mask_{id}.png')

