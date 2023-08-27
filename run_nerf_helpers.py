import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

        # the ML{'s design  mainly refers the Fig7 of the origin paper

        self.pts_linears=nn.ModuleList(
            [nn.Linear(input_ch,W)]+[nn.Linear(W,W) if i is not in self.skips else nn.Linear(W+input_ch,W) for i in range(D-1)]
        ) # define 0-7 layer and concat in the fifth layer

        self.view_linears=nn.ModuleList([nn.Linear(input_ch_views+W,W//2)])

        if use_viewdirs:
            self.feature_linear=nn.Linear(W,W)
            self.alpha_linear=nn.Linear(W,1)
            self.rgb_linear=nn.Linear(W//2,3)
        else:
            self.output_linear=nn.Linear(W,output_ch)

    def forward(self,x):
        input_pts,input_views=torch.split(x,[self.input_ch,self.input_ch_views],-1)
        h=input_pts
        for i ,l in enumerate(self.pts_linears):
            h=self.pts_linears[i](h)  # what?
            h=F.relu(h)
            if i in self.skips:
                h=torch.concat([input_pts,h],-1)

        if self.use_viewdirs:
            alpha=self.alpha_linear(h)
            feature=self.feature_linear(h)
            h=torch.concat([feature,input_views],-1)

            for i,l in enumerate(self.view_linears):
                h=self.view_linears[i](h)
                h=F.relu(h)

            rgb=self.rgb_linear(h)
            outputs=torch.concat([rgb,alpha],-1)
        else:
            outputs=self.output_linear(h)

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
        self.view_linears[0].bias,data=torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rgb_linear=2*self.D+4
        self.rgb_linear.weight.data=torch.from_numpy(np.transpose(weights[idx_rgb_linear]))
        self.rgb_linear.bias.data=torch.from_numpy(np.transpose(weights[idx_rgb_linear+1]))

        # Load alpha_linear
        idx_alpha_linear=2*self.D+6
        self.alpha_linear.weight.data=torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data=torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
