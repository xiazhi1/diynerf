import torch
import numpy as np
import  os,sys
import json
import imageio
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

    basedir=args.basedir
    expname=args.expname
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
    global_step=start

    bds_dict={
        'near':near,
        'far':far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # moving testing data to GPU
    render_poses=torch.Tensor(render_poses).to(device)

    # short circuit if only rendering out from trained model
    if args.render_only:
        print('render only')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images=images[i_test]
            else:
                # Default is smoother render_poses path
                images=None
            
            testsavedir=os.path.join(basedir,expname,'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path',start))
            os.makedirs(testsavedir,exist_ok=True)
            print('test poses shape',render_poses.shape)

            rgbs,_=render_path(render_poses,hwf,K,args.chunk,render_kwargs_test,gt_imgs=images,savedir=testsavedir,render_factor=args.render_factor)
            print('Done rendering',testsavedir)
            imageio.mimwrite(os.path.join(testsavedir,'video.mp4'),to8b(rgbs),fps=30,quality=8)






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
    grad_vars=list(model.parameters())

    model_fine=None
    if args.N_importance >0:
        model_fine=NeRF(D=args.netdepth_fine,W=args.netwidth_fine,
               input_ch=input_ch,output_ch=output_ch,skips=skips,
               input_ch_views=input_ch_views,use_viewdirs=args.use_viewdirs).to(device)
        grad_vars+=list(model_fine.parameters())

    network_query_fn=lambda inputs,viewdirs,network_fn:run_network(inputs,viewdirs,network_fn,
                                                                   embed_fn=embed_fn,
                                                                   embeddirs_fn=embeddirs_fn,
                                                                   netchunk=args.netchunk)
    
    # create optimizer
    optimizer=torch.optim.Adam(params=grad_vars,lr=args.lrate,betas=(0.9,0.999))
    start=0
    basedir=args.basedir
    expname=args.expname

    # load checkpoints
    if args.ft_path is not None and args.ft_path !='None':
        ckpts=[args.ft_path]
    else:
        ckpts=[os.path.join(basedir,expname,f) for f in sorted(os.listdir(os.path.join(basedir,expname))) if 'tar' in f]

    print('Found ckpts',ckpts)
    if len(ckpts)>0 and not args.no_reload:
        ckpt_path=ckpts[-1]
        print('Reloading from ',ckpt_path)
        ckpt=torch.load(ckpt_path)

        start=ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    

    render_kwargs_train={
        'network_query_fn':network_query_fn,
        'perturb':args.perturb,
        'N_importance':args.N_importance,
        'network_fn':model,
        'network_fine':model_fine,
        'N_samples':args.N_samples,
        'use_viewdirs':args.use_viewdirs,
        'white_bkgd':args.white_bkgd,
        'raw_noise_std':args.raw_noise_std,

    }

    render_kwargs_test={k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['preturb']=False
    render_kwargs_test['raw_noise_std']=0.

    return render_kwargs_train,render_kwargs_test,start,grad_vars,optimizer


def batchify(fn,chunk):
    # constructs a version of 'fn' that applies to smaller batches
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0,inputs.shape[0],chunk)],0)
    return ret

def run_network(inputs,viewdirs,fn,embed_fn,embeddirs_fn,netchunk=1024*64):
    # prepare inputs and applies network fn
    inputs_flat=torch.reshape(inputs,[-1,inputs.shape[-1]])
    embedded=embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs=viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat=torch.reshape(inputs_dirs,[-1,input_dirs.shape[-1]])
        embedded_dirs=embeddirs_fn(input_dirs_flat)
        embedded=torch.cat([embedded,embedded_dirs],-1)

    outputs_flat=batchify(fn,netchunk)(embedded)
    outputs=torch.reshape(outputs_flat,list(inputs.shape[:-1])+[outputs_flat.shape[-1]])
    return outputs


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
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--lrate",type=float,default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay",type=int,default=250,
                        help='exponential learning rate decay(in 1000 steps)')
    parser.add_argument("--ft_path",type=str,default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("no_reload",action='store_true',
                        help='do not reload weights from saved ckpt')



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
    parser.add_argument("--perturb",type=float,default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("raw_noise_std",type=float,default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only",action='store_true',
                        help='do not optimize,reload weights and render out render_poses path')
    parser.add_argument("--render_test",action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor",type=int,default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    return  parser

if __name__=='__main__':
    train()
