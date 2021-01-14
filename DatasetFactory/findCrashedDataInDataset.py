
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


root='/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/'

imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
tempmasklist=list(sorted(os.listdir(os.path.join(root, "masks"))))
counter = 0
for idx in range(len(imgs)):
    #print("idx : ", idx)
    # load images ad masks
    img_path = os.path.join(root, "images", imgs[idx])
    mask_path = os.path.join(root, "masks", tempmasklist[idx])
    img = Image.open(img_path).convert("RGB")
    # note that we haven't converted the mask to RGB,
    # because each color corresponds to a different instance
    # with 0 being background
    mask = Image.open(mask_path)

    mask = np.array(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]

    # get bounding box coordinates for each mask
    num_objs = len(obj_ids)
    boxes = []
    for i in range(num_objs):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes.append([xmin, ymin, xmax, ymax])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # there is only one class
    labels = torch.ones((num_objs,), dtype=torch.int64)
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    #print(img_path)
    
    image_id = torch.tensor([idx])
    try:
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    except :
        print(img_path)
        print(mask_path)
        os.remove(img_path)
        os.remove(mask_path)
        counter+=1
        continue

    for areaidx in area:       
        if areaidx < 1:
            print(img_path)
            print(mask_path)
            os.remove(img_path)
            os.remove(mask_path)
            counter+=1
            break
print(counter,' tane dosya klasör başına silindi')
#7901 Dosya var 