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

def main():
    args = ParserArguments()
    json_dict = dict()
    trainer = nnUNetTrainer(args.pkl,args.model_weights)

    A2C_img,A4C_img,A2C_mask,A4C_mask = load_dataset(datapath=args.data_root,trainer=trainer)


    for n,(img,mask) in enumerate(tqdm(zip(A2C_img,A2C_mask))):

        mask_prob = trainer.network(img)

        # sigmoid
        mask_prob_sig = torch.sigmoid(mask_prob)
        mask_prob_sig = mask_prob_sig[0][0].detach().numpy()
        mask_prob_sig = np.where(mask_prob_sig < 0.5, mask_prob_sig, 0)

        # negative
        mask_prob_np = mask_prob[0][0].detach().numpy()
        mask_prob_np = np.where(mask_prob_np < 0, mask_prob_np, 0)

        # print(mask_prob.shape, mask.shape)
        # print(mask_prob)
        # DICE & JACCARD

        dc_np = get_dice(mask, mask_prob_np)
        jc_np = get_jaccard(mask, mask_prob_np)

        dc_sig = get_dice(mask, mask_prob_sig)
        jc_sig = get_jaccard(mask, mask_prob_sig)

        id = 'A2C_' + str(n)
        json_dict[id] = {}

        # np: negative, sig = tensor sigmoid
        json_dict[id] = {'dice_np': dc_np, 'jaccard_np': jc_np, 'dice_sig': dc_sig, 'jaccard_sig': jc_sig}

        # export
        plt.imshow(img.detach().numpy()[0][0],cmap="gray")
        plt.imshow(mask_prob_np, alpha=.2)

        plt.savefig(f"exp/plots/A2C_{n}.png")

    # for n,(img,mask) in enumerate(tqdm(zip(A4C_img,A4C_mask))):

    #     mask_prob = trainer.network(img)
    #     mask_prob = mask_prob[0][0].detach().numpy()
    #     mask_prob = np.where(mask_prob < 0, mask_prob, 0)

    #     # DICE & JACCARD
    #     dc = get_dice(mask, mask_prob)
    #     jc = get_jaccard(mask, mask_prob)

    #     id = 'A4C_' + str(n)
    #     json_dict[id] = {}
    #     json_dict[id] = {'dice': dc, 'jaccard': jc}
        
    #     # export
    #     plt.imshow(img.detach().numpy()[0][0],cmap="gray")
    #     plt.imshow(mask_prob, alpha=.2)

    #     plt.savefig(f"exp/plots/A4C_{n}.png")

    # print(json_dict)
    # mean_A2C, mean_A4C = compute_mean(json_dict)
    mean_A2C= compute_mean(json_dict)
    json_dict['mean_A2C'] = mean_A2C
    # json_dict['mean_A4C'] = mean_A4C

    write_json(json_dict, './exp/result_A2C.json')


if __name__ == '__main__':
	main()

""" 
output은 label 파일과 동일하게 npy 형식으로 나와야함. 
Validation의 성능 평가는 실제 평가의 참고용 점수로 활용됩니다.

제출하실 때 성능 평가가 가능한 test code를 함께 제출해주시기 바랍니다.

평가는 제출하신 평가 코드로 진행하며, output은 label 파일과 동일한 형식이어야 합니다.
####################################################

A2C, A4C 이미지 모두 학습 시킨 모델 1개 또는 별도로 학습시킨 모델 2개중
체적일치도(Dice Similarity Coefficient, DSC), 유사성측도(Jaccard index, JI)이 높은 것을 제출 부탁드리겠습니다.

하나의 공통된 모델을 개발하거나 두개의 모델을 별도로 개발하여도 무방하나,
모델의 성능은 각각의 A2C, A4C 데이터에 대하여 평가됩니다.
==========================================================> 두 개 모델을 제출하면 A4C 모델도 A2C에 대해 평가????????
"""