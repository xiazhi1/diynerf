import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# misc
img2mse=lambda x, y :torch.mean((x-y) ** 2)
mse2psnr=lambda x:-10.*torch.log(x)/torch.log(torch.Tensor([10.]))
to8b=lambda x:(255*np.clip(x,0,1)).astype(np.uint8)

# call PE
def get_embedder(multires,i=0):
    if i==-1:
        return nn.Identity(),3
    embedder_kwargs={
        'include_input':True,
        'input_dims':3,
        'max_freq_log2':multires-1,
        'num_freq':multires,
        'log_sampling':True,
        'periodic_fns':[torch.sin,torch.cos],

    }
    embedder_obj=Embedder(**embedder_kwargs)
    embed= lambda x,eo=embedder_obj:eo.embed(x)
    return embed,embedder_obj.out_dim

# PE
class Embedder:
    def __init__(self,**kwargs):
        self.kwargs=kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns=[]
        d=self.kwargs['input_dims']
        out_dim=0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x:x)
            out_dim+=d
        max_freq=self.kwargs['max_freq_log2']
        N_freqs=self.kwargs['num_freq']

        if self.kwargs['log_sampling']:
            freq_bands=2.**torch.linspace(0.,max_freq,steps=N_freqs)
        else:
            freq_bands=torch.linspace(2.**0.,2.**max_freq,steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x,p_fn=p_fn,freq=freq:p_fn(x * freq))
                out_dim+=d



        self.embed_fns=embed_fns
        self.out_dim=out_dim

    def embed(self,inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns],-1)








# model
class NeRF(nn.Module):
    def __init__(self,D=8,W=256,input_ch=3,input_ch_views=3,output_ch=4,skips=[4],use_viewdirs=False):
        super(NeRF,self).__init__()
        self.D=D
        self.W=W
        self.input_ch=input_ch
        self.input_ch_views=input_ch_views
        self.skips=skips # skips is used to decide where to add the input_ch
        self.use_viewdirs=use_viewdirs

        # the MLP's design  mainly refers the Fig7 of the origin paper

        self.pts_linears=nn.ModuleList(
            [nn.Linear(input_ch,W)]+[nn.Linear(W,W) if i not in self.skips else nn.Linear(W+input_ch,W) for i in range(D-1)]
        ) # define 0-7 layer and concat in the fifth layer

        self.view_linears=nn.ModuleList([nn.Linear(input_ch_views+W,W//2)])

        if use_viewdirs:
            self.feature_linear=nn.Linear(W,W)
            self.alpha_linear=nn.Linear(W,1)
            self.rgb_linear=nn.Linear(W//2,3)
        else:
            self.output_linear=nn.Linear(W,output_ch)

    def forward(self,x):
        input_pts,input_views=torch.split(x,[self.input_ch,self.input_ch_views],dim=-1)
        h=input_pts
        for i ,l in enumerate(self.pts_linears):
            h=self.pts_linears[i](h)  # what?
            h=F.relu(h)
            if i in self.skips:
                h=torch.cat([input_pts,h],-1)

        if self.use_viewdirs:
            alpha=self.alpha_linear(h)
            feature=self.feature_linear(h)
            h=torch.cat([feature,input_views],-1)

            for i,l in enumerate(self.view_linears):
                h=self.view_linears[i](h)
                h=F.relu(h)

            rgb=self.rgb_linear(h)
            outputs=torch.cat([rgb,alpha],-1)
        else:
            outputs=self.output_linear(h)
        
        return outputs

    def load_weights_from_keras(self,weights): # use it to load weights and bias of the MLP
        assert self.use_viewdirs,"Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears=2*i
            self.pts_linears[i].weight.data=torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data=torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear=2*self.D
        self.feature_linear.weight.data=torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data=torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears=2*self.D+2
        self.view_linears[0].weight.data=torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.view_linears[0].bias.data=torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rgb_linear=2*self.D+4
        self.rgb_linear.weight.data=torch.from_numpy(np.transpose(weights[idx_rgb_linear]))
        self.rgb_linear.bias.data=torch.from_numpy(np.transpose(weights[idx_rgb_linear+1]))

        # Load alpha_linear
        idx_alpha_linear=2*self.D+6
        self.alpha_linear.weight.data=torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data=torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


# Ray helpers
def get_rays(H,W,K,c2w):
    i,j=torch.meshgrid(torch.linspace(0,W-1,W),torch.linspace(0,H-1,H)) # pytorch's meshfgrid has indexing = 'ij'
    i=i.t()
    j=j.t() # what is the function of t()?
    dirs=torch.stack([(i-K[0][2])/K[0][0],-(j-K[1][2])/K[1][1],-torch.ones_like(i)],-1)
    # Rotate ray directions from camera frame to the world frame
    rays_d=torch.sum(dirs[...,np.newaxis,:] * c2w[:3,:3],-1) # dot product , equals to :[c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame , it is the origin of all rays
    rays_o=c2w[:3,-1].expand(rays_d.shape)
    return rays_o,rays_d

def get_rays_np(H,W,K,c2w):
    i,j=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32),indexing='xy')
    dirs=np.stack([(i-K[0][2])/K[0][0],-(j-K[1][2])/K[1][1],-np.ones_like(i)],-1)
    # Rotate ray directions from camera frame to the world frame
    rays_d=np.sum(dirs[...,np.newaxis,:] * c2w[:3,:3],-1) # dot product , equals to :[c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame , it is the origin of all rays
    rays_o=np.broadcast_to(c2w[:3,-1],np.shape(rays_d))
    return rays_o,rays_d

def ndc_rays(H,W,focal,near,rays_o,rays_d):
    # shift ray origins to near plane
    t=-(near+rays_o[...,2])/rays_d[...,2]
    rays_o=rays_o+t[...,None]*rays_d

    # projectiion
    o0=-1./(W/(2.*focal)) *rays_o[...,0]/rays_o[...,2]
    o1=-1./(H/(2.*focal)) *rays_o[...,1]/rays_o[...,2]
    o2=1.+2.*near/rays_o[...,2]

    d0=-1./(W/(2.*focal))*(rays_d[...,0]/rays_d[...,2]-rays_o[...,0]/rays_o[...,2])
    d1=-1./(H/(2.*focal))*(rays_d[...,1]/rays_d[...,2]-rays_o[...,1]/rays_o[...,2])
    d2=-2.*near/rays_o[...,2]

    rays_o=torch.stack([o0,o1,o2],-1)
    rays_d=torch.stack([d0,d1,d2],-1)

    return rays_o,rays_d

def sample_pdf(bins,weights,N_samples,det=False,pytest=False):
    # get pdf
    weights=weights + 1e-5 #prevent nans
    pdf = weights /torch.sum(weights,-1,keepdim=True)
    cdf = torch.cumsum(pdf,-1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],-1) # (batch,len(bins))

    # Take uniform samples
    if det:
        u=torch.linspace(0.,1.,steps=N_samples)
        u=u.expand(list(cdf.shape[:-1])+[N_samples])
    else:
        u=torch.rand(list(cdf.shape[:-1])+[N_samples])

    # pytest,overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape=list(cdf.shape[:-1])+[N_samples]
        if det:
            u=np.linspace(0.,1.,N_samples)
            u=np.broadcast_to(u,new_shape)
        else:
            u=np.random.rand(*new_shape)
        u=torch.Tensor(u)

    # Invert CDF
    u=u.contiguous()
    inds=torch.searchsorted(cdf,u,right=True)
    below=torch.max(torch.zeros_like(inds-1),inds-1)
    above=torch.min((cdf.shape[-1]-1)*torch.ones_like(inds),inds)
    inds_g=torch.stack([below,above],-1) #(batch,N_samples,2)

    matched_shape=[inds_g.shape[0],inds_g.shape[1],cdf.shape[-1]]
    cdf_g=torch.gather(cdf.unsqueeze(1).expand(matched_shape),2,inds_g)
    bins_g=torch.gather(bins.unsqueeze(1).expand(matched_shape),2,inds_g)

    denom=(cdf_g[...,1]-cdf_g[...,0])
    denom=torch.where(denom<1e-5,torch.ones_like(denom),denom)
    t=(u-cdf_g[...,0])/denom
    samples=bins_g[...,0]+t*(bins_g[...,1]-bins_g[...,0])

    return samples


