from glob import glob
import os
import numpy as np
import torch

from utils.transform import *

def load_dataset(datapath,trainer,args):
    imglist = []
    masklist = []
    labellist = []
    origlist = []
    positionlist = []

    for n,img_path in enumerate(sorted(glob(os.path.join(datapath,args.ptype,"*.png")))):
        imgname = os.path.basename(img_path).split(".")[0]
        labellist.append(imgname)

        img = cv2.imread(img_path, 0)
        original_size = img.shape
        origlist.append(img.copy())

        img,crop_position = center_crop(img, trainer.patch_size[0],trainer.patch_size[1])
        positionlist.append(crop_position)
        img = image_minmax(img)
        img = cv2.resize(img, ( trainer.patch_size[1],trainer.patch_size[0]))
        # Add channel axis
        img = img[None,None, ...].astype(np.float32)
        
        img_torch = torch.from_numpy(img)
        imglist.append(img_torch)

        mask = np.load(img_path.replace(".png",".npy"))

        mask,_ = center_crop(mask, trainer.patch_size[0],trainer.patch_size[1])
        mask = cv2.resize(mask, ( trainer.patch_size[1],trainer.patch_size[0]))

        masklist.append(mask)

    print(f"load {len(imglist)} {args.ptype} images!")


    return imglist,masklist,labellist,origlist,positionlist
