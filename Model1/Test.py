from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import pandas as pd
import cv2
import os
import numpy as np
import torch

test = pd.read_csv("test_data.csv")

sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
model_path = "model.torch"
num_samples = 30  

def read_image(image_path): 
    img = cv2.imread(image_path)[..., ::-1]  
    mask = np.ones(img.shape)

    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)
    
    return img, mask

def get_points(mask, num_points): 
    points = []
    coords = np.argwhere(mask > 0)
    for i in range(num_points):
        yx = np.array(coords[np.random.randint(len(coords))])
        points.append([[yx[1], yx[0]]])
    return np.array(points)

model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)    


output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

predictor.model.load_state_dict(torch.load(model_path))
iou_values_0 = []  
iou_values_1 = []  

for j in range(len(test)):
    image_path = test.iloc[j]["image"]
    annotation_path = test.iloc[j]["annotation"]
    ann_map = cv2.imread(test.iloc[j]["Mask"])[...,::-1]

    ann_map = cv2.resize(ann_map, (1024,1024),interpolation=cv2.INTER_NEAREST)
    red_channel = ann_map[..., 0]

    _, gt_mask = cv2.threshold(red_channel, 1, 255, cv2.THRESH_BINARY)

    image, mask = read_image(image_path)
    input_points = get_points(mask, num_samples)

    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=np.ones([input_points.shape[0], 1])
        )

    masks = masks[:, 0].astype(bool)
    sorted_masks = masks[np.argsort(scores[:, 0])][::-1].astype(bool)

    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)

    for i in range(sorted_masks.shape[0]):
        mask = sorted_masks[i]
        
        if mask.sum() == 0:
            continue
        
        if (mask * occupancy_mask).sum() / mask.sum() > 0.15:
            continue
        
        mask[occupancy_mask] = 0
        seg_map[mask] = i + 1
        occupancy_mask[mask] = 1

    rgb_image = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for id_class in range(1, seg_map.max() + 1):
        rgb_image[seg_map == id_class] = [255, 0, 0]
    
    mix_image = ((rgb_image / 2 + image / 2).astype(np.uint8))

    mix_image_name = os.path.join(output_dir, f"tahmin{j}.png")
    
    prd_mask_0 = (sorted_masks[0] == 0).astype(np.float32)  
    gt_mask_0 = (gt_mask == 0).astype(np.float32)  
    
    intersection_0 = np.logical_and(gt_mask_0, prd_mask_0).sum()
    union_0 = np.logical_or(gt_mask_0, prd_mask_0).sum()
    
    if union_0 == 0:
        iou_0 = 0.0
    else:
        iou_0 = intersection_0 / union_0
    
    if iou_0 > 0.0:  
        iou_values_0.append(iou_0)  
    
    prd_mask_1 = (sorted_masks[0] == 1).astype(np.float32)  
    gt_mask_1 = (gt_mask == 255).astype(np.float32)  
    
    intersection_1 = np.logical_and(gt_mask_1, prd_mask_1).sum()
    union_1 = np.logical_or(gt_mask_1, prd_mask_1).sum()
    
    if union_1 == 0:
        iou_1 = 0.0
    else:
        iou_1 = intersection_1 / union_1
    
    if iou_1 > 0.0:  
        iou_values_1.append(iou_1)  
    
    
    print(f"Image {j}: IoU for label 0 = {iou_0}, IoU for label 1 = {iou_1}")

    
average_iou_0 = sum(iou_values_0) / len(iou_values_0)
average_iou_1 = sum(iou_values_1) / len(iou_values_1)
print(len(iou_values_0))
print(len(iou_values_1))
print(f"Average IoU for label 0: {average_iou_0}")
print(f"Average IoU for label 1: {average_iou_1}")
