import torch
import numpy
import  os,sys
import json

import configargparse

from load_blender import load_blender_data





def train ():

    parser=config_parser()
    args=parser.parse_args()
    print(args)

    # load data from dataset
    if args.dataset_type=='blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split







def config_parser():

    # init options
    parser=configargparse.ArgumentParser(description='add parameters for the model')
    parser.add_argument('--config',is_config_file=True, help='config file path')
    parser.add_argument('--expname',type=str, help='experiment name')
    parser.add_argument('--basedir',type=str, default='./logs/',help='where to store logs and ckpts')
    parser.add_argument('--datadir',type=str, default='./data/nerf_synthetic/lego',help='input data directory')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='now only support blender')
    parser.add_argument('--testskip', type=int, default=8,
                        help="use to load 1/N images from test/val sets")

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_false',
                        help='load blender synthetic data at 400x400 instead of 800x800')


    return  parser

if __name__=='__main__':
    train()
