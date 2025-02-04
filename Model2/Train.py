import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def read_batch(data): 


    Img = cv2.imread(data["image"])[...,::-1]  
    ann_map = cv2.imread(data["Mask"])[...,::-1]

    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) 
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)
    red_channel = ann_map[..., 0]

    _, binary_red = cv2.threshold(red_channel, 1, 255, cv2.THRESH_BINARY)
    plt.imshow(binary_red)
    plt.show()
    inds = np.unique(binary_red)[1:] 
    points= []
    masks = []
    for ind in inds:
        mask=(binary_red == ind).astype(np.uint8) 
        masks.append(mask)
        coords = np.argwhere(mask > 0) 
        yx = np.array(coords[np.random.randint(len(coords))]) 
        points.append([[yx[1], yx[0]]])
    return Img,np.array(masks),np.array(points), np.ones([len(masks),1])

sam2_checkpoint = "checkpoints/sam2_hiera_small.pt" 
model_cfg = "sam2_hiera_s.yaml"
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") 
predictor = SAM2ImagePredictor(sam2_model)


predictor.model.sam_mask_decoder.train(True) 
predictor.model.sam_prompt_encoder.train(True) 

predictor.model.image_encoder.train(True)

optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler() 

train_data = pd.read_csv("train_data.csv")
iou_values = []
for itr in range(100000):
    with torch.cuda.amp.autocast(): 
        i = np.random.choice(len(train_data))
        image,mask,input_point, input_label = read_batch(train_data.iloc[i]) 
    
        if mask.shape[0]==0: continue   
        predictor.set_image(image)

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

        batched_mode = unnorm_coords.shape[0] > 1 
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features,)
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])


        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() 

        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss=seg_loss+score_loss*0.05  


        predictor.model.zero_grad() 
        scaler.scale(loss).backward()  
        scaler.step(optimizer)
        scaler.update() 

        if itr%1000==0: torch.save(predictor.model.state_dict(), "model.torch");print("save model")


        if itr==0: mean_iou=0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        iou_values.append(mean_iou)
        print("step)",itr, "Accuracy(IOU)=",mean_iou)

plt.plot(range(len(iou_values)), iou_values)
plt.xlabel("Iteration")
plt.ylabel("Mean IOU")
plt.title("IOU vs Iteration")
plt.show()