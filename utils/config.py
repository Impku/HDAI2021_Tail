import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ParserArguments(ptype):
    args = argparse.ArgumentParser()

    # Directory Setting 
    args.add_argument('--data_root', type=str, default='./data/', help='dataset directory')
    args.add_argument('--exp', type=str, default="exp/", help='Folder path to save npy')
    args.add_argument('--plot_png', type=str2bool,default="false", help='Save to png?')
    args.add_argument('--json', type=str2bool,default="false", help='Save to png?')

    args = args.parse_args()

    args.ptype = ptype

    if args.ptype == "A2C":
        args.pkl1 = "weights/model1_A2C.pkl"
        args.pkl2 = "weights/model2_A2C.pkl"
        args.pkl3 = "weights/model3_A2C.pkl"

        args.model_weights1 = "weights/model1_A2C.model"
        args.model_weights2 = "weights/model2_A2C.model"
        args.model_weights3 = "weights/model3_A2C.model"

        args.exp = os.path.join(args.exp,"A2C")

    if args.ptype == "A4C":
        args.pkl1 = "weights/model1_A4C.pkl"
        args.pkl2 = "weights/model2_A4C.pkl"
        args.pkl3 = "weights/model3_A4C.pkl"

        args.model_weights1 = "weights/model1_A4C.model"
        args.model_weights2 = "weights/model2_A4C.model"
        args.model_weights3 = "weights/model3_A4C.model"

        args.exp = os.path.join(args.exp,"A4C")

    print("\n")
    print("------------Parameters--------------")
    print(f"- Image type: {args.ptype}")
    print(f"- Data root: {args.data_root}")
    print(f"- Export folder: {args.exp}")
    print(f"- Export PNG files?: {args.plot_png}")
    print(f"- Export Json file?: {args.json}")
    print("------------Parameters--------------")
    print("\n")

    return args
