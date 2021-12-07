from matplotlib import cm
import torch
import SimpleITK as sitk
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.transform import *
from utils.config import *
from net.model import *
from dataset.dataloader import * 
from utils.metric import *
from utils.file_op import *

from tqdm import tqdm

def recovery_mask(mask_prob_sig,orig,position):

    new_mask = np.zeros_like(orig)

    mask_prob_sig = (mask_prob_sig * 255).astype(np.uint8)
    mask_prob_sig = cv2.resize(mask_prob_sig,(position[3]-position[2],position[1]-position[0]))

    new_mask[position[0]:position[1],position[2]:position[3]] = mask_prob_sig

    return new_mask


def main():
    args = ParserArguments(ptype="A4C")
    
    json_dict = dict()

    trainer1 = nnUNetTrainer(args.pkl1,args.model_weights1)
    trainer2 = nnUNetTrainer(args.pkl2,args.model_weights2)
    trainer3 = nnUNetTrainer(args.pkl3,args.model_weights3)
    print(" - Load trained models")


    print(" - Prepare test images")
    A4C_img,A4C_mask,A4C_label,A4C_orig,A4C_position = load_dataset(datapath=args.data_root,trainer=trainer1,args=args)


    for n,(img,mask,name,orig,position) in enumerate(tqdm(zip(A4C_img,A4C_mask,A4C_label,A4C_orig,A4C_position))):

        mask_prob1 = trainer1.network(img)
        mask_prob2 = trainer2.network(img)
        mask_prob3 = trainer3.network(img)

        # aggregation
        mask_prob = torch.stack([mask_prob1,mask_prob2,mask_prob3],dim=-1)
        # mean
        # mask_prob = torch.mean(mask_prob,dim=-1)
        # max
        # mask_prob = torch.max(mask_prob,dim=-1).values
        # min
        mask_prob = torch.min(mask_prob,dim=-1).values
        # median
        # mask_prob = torch.median(mask_prob,dim=-1).values

        # sigmoid
        mask_prob_sig = torch.sigmoid(mask_prob)
        mask_prob_sig = mask_prob_sig[0][0].detach().numpy()
        mask_prob_sig = np.where(mask_prob_sig < 0.5, 1, 0)

        dc_sig = get_dice(mask, mask_prob_sig)
        jc_sig = get_jaccard(mask, mask_prob_sig)

        # id = 'A4C_' + str(n)
        json_dict[name] = {}

        # np: negative, sig = tensor sigmoid
        json_dict[name] = {'dice_sig': dc_sig, 'jaccard_sig': jc_sig}

        print(name,dc_sig,jc_sig)
        mask_orig = recovery_mask(mask_prob_sig,orig,position)
        mask_orig = np.where(mask_orig < 1, 0, 1)

        if args.plot_png == True:
            # export
            plt.imshow(orig,cmap="gray")
            plt.imshow(mask_orig, alpha=.2)
            plt.axis("off")
            plt.savefig(os.path.join(args.exp,f"{name}.png"),bbox_inches="tight",pad_inches=0,dpi=100)

        np.save(os.path.join(args.exp,f"{name}.npy"),mask_orig.astype(np.uint8))


    mean_dice, mean_jaccard = compute_mean(json_dict, option='A4C')
    json_dict[f'mean_dice_{args.ptype}'] = mean_dice
    json_dict[f'mean_jaccard_{args.ptype}'] = mean_jaccard

    if args.json == True:
        write_json(json_dict, f'./exp/result_{args.ptype}.json')


if __name__ == '__main__':
	main()
