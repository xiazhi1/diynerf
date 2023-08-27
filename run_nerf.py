import torch
import numpy as np
import  os,sys
import json

import configargparse

from load_blender import load_blender_data
from run_nerf_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
# it is a test for git

def train ():

    parser=config_parser()
    args=parser.parse_args()
    print(args)

    K =None # K is the camera intenral paremeter matrix
    # load data from dataset
    if args.dataset_type=='blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near=2.
        far=6.

        if args.white_bkgd :
            images=images[...,:3]*images[...,-1:]+(1.-images[...,-1:])
        else:
            images=images[...,:3]

    # Cast intrinsics to right types

    H,W,focal=hwf
    H,W=int(H),int(W)
    hwf=[H,W,focal]

    if K is None:
        K=np.array([
            [focal,0,0.5*W],
            [0,focal,0.5*H],
            [0,0,1]
        ]) # camera intristics

    # create log folder when debug is not used

    # basedir=args.basedir
    # expname=args.expname
    # os.makedirs(os.path.join(basedir,expname),exist_ok=True)
    # f=os.path.join(basedir,expname,'args.txt')
    # with open(f , 'w') as file:
    #     for arg in sorted(vars(args)):
    #         attr=getattr(args,arg)
    #         file.write('{}={}\n'.format(args,attr))
    # if args.config is not None:
    #     f = os.path.join(basedir, expname, 'config.txt')
    #     with open(f, 'w') as file:
    #         file.write(open(args.config, 'r').read())

    # create nerf model
    render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer=create_nerf(args)

# Instantiate NeRF's MLP model.
def create_nerf(args):
    embed_fn,input_ch=get_embedder(args.multires,args.i_embed)
    input_ch_views=0
    embeddirs_fn=None
    if args.use_viewdirs:
        embeddirs_fn,input_ch_views=get_embedder(args.multires_views,args.i_embed)
    output_ch=5 if args.N_importance>0 else 4
    skips=[4]
    model=NeRF(D=args.netdepth,W=args.netwidth,
               input_ch=input_ch,output_ch=output_ch,skips=skips,
               input_ch_views=input_ch_views,use_viewdirs=args.use_viewdirs).to(device)



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

    # train options
    parser.add_argument("--net_depth",type=int,default=8,
                        help="layers in network")
    parser.add_argument("--netwidth",type=int,default=256,
                        help="channels per layer")
    parser.add_argument("netdepth_fine",type=int,default=8,
                        help="layers in fine network")
    parser.add_argument("--nerwidth_fine",type=int,default=256,
                        help="channels [er layer in fine network")

    # render options
    parser.add_argument("--multires",type=int,default=10,
                        help="log2 of max frequency of positional encoding(position)")
    parser.add_argument("--i_embed",type=int,default=0,
                        help="set 0 for default positional encoding, -1 for none")
    parser.add_argument("--use_viewdirs",action='store_True',
                        help="use full 5D input rather than 3D")
    parser.add_argument("--multires_views",type=int,default=4,
                        help="log2 of max frequency of positional encoding(direction)")
    parser.add_argument("--N_importance",type=int,default=0,
                        help="number of additional fine samples per ray")
    parser.add_argument("--N_samples",type=int,default=64,
                        help="number of coarse samples per ray")
    return  parser

if __name__=='__main__':
    train()
