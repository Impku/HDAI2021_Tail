from json import load
import numpy as np
from medpy import metric
from utils.file_op import *

# we assume that data shapes are matching for ref, mask

def get_dice(gt, pred):

    tp, tn, fp, fn = get_confusion_matrix(gt, pred)
    
    return float(2. * tp/(2*tp+fp+fn))

def get_jaccard(gt, pred):

    tp, tn, fp, fn = get_confusion_matrix(gt, pred)
    
    return float(tp/(tp+fp+fn)) 

def get_confusion_matrix(gt, pred):

    tp = int(((pred != 0) * (gt != 0)).sum())
    tn = int(((pred == 0) * (gt == 0)).sum())
    fp = int(((pred != 0) * (gt == 0)).sum())
    fn = int(((pred == 0) * (gt != 0)).sum())

    return tp, tn, fp, fn

def compute_mean(file: dict, option):

    mean_dice_A2C = sum([v['dice_sig'] for k,v in file.items()])/100
    mean_jaccard_A4C = sum([v['jaccard_sig'] for k,v in file.items()])/100

    return mean_dice_A2C, mean_jaccard_A4C