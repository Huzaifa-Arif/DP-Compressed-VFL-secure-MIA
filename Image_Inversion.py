
import torch
import copy
from mvcnn_bottom_small2 import *
import pickle
#import torch.nn as nn
#from quant import *
import math
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from scipy.linalg import norm
from random import choices
from scipy.stats import binom
import os

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from PIL import Image


import argparse
import sys
import traceback

class RoundGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
    @staticmethod
    def backward(ctx, g):
        return g
class FloorGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()
    @staticmethod
    def backward(ctx, g):
        return g
class CeilGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.ceil()
    @staticmethod
    def backward(ctx, g):
        return g

def topk(tensor, compress_ratio):
    """
    Get topk elements in tensor
    """
    shape = tensor.shape
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    numel = tensor.numel()
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed.view(shape)

def quantize_scalar(x, quant_min=0, quant_max=1, quant_level=5):
   
    """Uniform quantization approach

    Notebook: C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original signal
        quant_min: Minimum quantization level
        quant_max: Maximum quantization level
        quant_level: Number of quantization levels

    Returns:
        x_quant: Quantized signal
    """
    x_op = x.detach().clone().numpy()
    x_np = (x_op.flatten())
    
    ## Move into 0,1 range:
    x_normalize = torch.divide(x,torch.max(x).item())
    x_normalize = torch.nan_to_num(x_normalize)
  
    
    
   
    ## Move out of 0,1 range :
    x_normalize = torch.divide(torch.multiply((x_normalize-quant_min),torch.tensor(quant_level - 1)),(quant_max-quant_min)) 
    
     
      
  
    dither = np.random.uniform(-(quant_max-quant_min)/(2*(quant_level-1)),
				(quant_max-quant_min)/(2*(quant_level-1)),
				size=x_normalize.shape) 
    dither = torch.from_numpy(dither)
            
    #dither =  0 # We are using a non-dithered quantization
    x_normalize = x_normalize + dither

    ################### Deterministic Change ############################# 
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = x_normalize
    x_normalize_quant = RoundGradient.apply(x_normalize)
    
 

    
  
    
    ################## Stochastic Change ###############################
    # temp = x_normalize.view(x_np.shape)
    
    # for i in range(len(temp)):
    #     x_s = FloorGradient.apply(temp[i])
    #     x_l = CeilGradient.apply(temp[i])
    #     p = temp[i] - x_s
    #     temp[i] = choices( list([x_s,x_l]), list([1-p,p]))[0]
    # x_normalize_quant = temp.view(x_op.shape)
    
    
    ################### Stochastic New Implementation ###################
    
    ## Choosing parameters of the Binomial Distribution
    eps =  4.0
    delta = 10**(-9)
    a1 = math.log(2/delta) *  8/pow(eps,2)
    print(a1)
    N = int(math.ceil(a1)* 1) ## I have made it a multiple of 2 deliberatly
    p_b = 0.5
    num_clients = 3
    m = N // num_clients
    #print(m,N)
    ## Generating Binomial PMF
  
    r_values = list(range(m+1))
    # list of pmf values
    dist = [binom.pmf(r, m, p_b) for r in r_values ]
    ## Binomial Variable chosen
    
    Z = torch.ones(x_normalize_quant.shape)
    for i in range(len(Z)):
        Z[i] = choices(r_values, dist)[0]
        
    # At server add the binomial noise
    x_normalize_quant = x_normalize_quant # + Z #- int(torch.round(torch.mean(Z))) 
       
    x_quant = (x_normalize_quant) * (quant_max-quant_min)/(quant_level -1) + quant_min
    #x_quant = torch.divide(x_normalize_quant,a)
    
    ## Move out of 0,1 range:
    #dither = 0    
    x_quant = torch.multiply(x_quant - dither,torch.max(x).item())
    #x_quant.to_numpy()
    
   
    

    
    return x_quant

def alpha_prior(x, alpha=2.):
    return torch.abs(x.view(-1)**alpha).sum()
    
def tv_norm(x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = img - img # set size of derivative and set border = 0
    dx = img - img
    dy[:,1:,:] = -img[:,:-1,:] + img[:,1:,:]
    dx[:,:,1:] = -img[:,:,:-1] + img[:,:,1:]
    return ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum()


def norm_loss(input, target):
    return torch.div(alpha_prior(input - target, alpha=2.), alpha_prior(target, alpha=2.))


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Clip(object):
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t>1] = 1
        t[t<0] = 0
        return t


#function to decay the learning rate
def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor 


def get_pytorch_module(net, blob):
    modules = blob.split('.')
    if len(modules) == 1:
        return net._modules.get(blob)
    else:
        curr_m = net
        for m in modules:
            curr_m = curr_m._modules.get(m)
        return curr_m

 #network='alexnet'
def invert(filename = 'Comp=0.01.pt',image = 'chair.png',size=224,network='alexnet', layer='features.6', alpha=6, beta=2, 
        alpha_lambda=1e-5,  tv_lambda=1e-5, epochs=200, learning_rate=1e2, 
        momentum=0.9, decay_iter=100, decay_factor=1e-1, print_iter=25, 
        cuda=False,comp=0.01):

    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        #transforms.Scale(size=size),
        transforms.CenterCrop(size=500),
        transforms.Resize(224),
        transforms.ToTensor()
        #transforms.Normalize(mu, sigma),
    ])

    detransform = transforms.Compose([
        #Denormalize(mu, sigma),
        Clip(),
        transforms.ToPILImage(),
    ])

    #model = models.__dict__[network](pretrained=True)
    model=mvcnn_bottom(pretrained=False)
    #import pdb;pdb.set_trace()
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if cuda:
        model.cuda()
   
    img_ = transform(Image.open(image).convert("RGB")).unsqueeze(0)
    #f, ax = plt.subplots(1,2)
    #ax[0].imshow(detransform(img_[0]))
    #print(img_.size())

    activations = []

    def hook_acts(module, input, output):
        activations.append(output)

    def get_acts(model, input): 
        del activations[:]
        _ = model(input)
        assert(len(activations) == 1)
        return activations[0]

    _ = get_pytorch_module(model, layer).register_forward_hook(hook_acts)
    
    input_var = Variable(img_.cuda() if cuda else img_).reshape(1,1,3,size,size)
    #input_var = quantize_scalar(input_var_0)
    ref_acts = get_acts(model, input_var).detach()

    x_ = Variable((1e-3 * torch.randn(*img_.size()).cuda().reshape(1,1,3,size,size) if cuda else 
        1e-3 * torch.randn(*img_.size()).reshape(1,1,3,size,size)), requires_grad=True)
    

    alpha_f = lambda x: alpha_prior(x, alpha=alpha)
    tv_f = lambda x: tv_norm(x, beta=beta)
    loss_f = lambda x: norm_loss(x, ref_acts)

    optimizer = torch.optim.SGD([x_], lr=learning_rate, momentum=momentum)
 
    
   
    for i in range(epochs):
        #import pdb;pdb.set_trace()
        acts1 = get_acts(model, x_)
        #acts = topk(acts_1,comp)
        acts = quantize_scalar(acts1,quant_level = comp)
        #acts = acts1 ## No compression
        #import pdb;pdb.set_trace()
        #print(i,"Iteration")

        alpha_term = alpha_f(x_)
        tv_term = tv_f(x_)
        loss_term = loss_f(acts)

        tot_loss = alpha_lambda*alpha_term + tv_lambda*tv_term + loss_term
        
        #if (i+1) % print_iter == 0:
            #print(tot_loss)
        #     print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f' % (i+1,
        #         alpha_term.data.cpu().numpy()[0], tv_term.data.cpu().numpy()[0],
        #         loss_term.data.cpu().numpy()[0], tot_loss.data.cpu().numpy()[0]))

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        if (i+1) % decay_iter == 0:
            decay_lr(optimizer, decay_factor)

    f, ax = plt.subplots(1,2)

    ax[0].imshow(detransform(img_[0]))
    ax[1].imshow(detransform(x_[0,0].data.cpu()))
    for a in ax:
          a.set_xticks([])
          a.set_yticks([])
        
    #import pdb; pdb.set_trace() torch.rand(2, 3)
    
    error = torch.mean(torch.linalg.norm(img_[0] -  x_[0,0] ,dim=(1,2), ord='fro' ) )
    noise_error = torch.mean(torch.linalg.norm(img_[0] -  torch.randn(img_[0].size()) ,dim=(1,2), ord='fro' ) )
    plt.show()
    return error,noise_error 


if __name__ == '__main__':
   

   
   try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', type=str,default='chair.png') 
        parser.add_argument('--network', type=str, default='alexnet')
        parser.add_argument('--size', type=int, default=224)
        parser.add_argument('--layer', type=str, default='features.4')
        ## To change ####
        ## Vary this between 1 and 12 ###
        #parser.add_argument('--filename', type=str,default='Comp=1.00.pt')
        #parser.add_argument('--compression', type=float, default=1.00)
        ## To change ####
        
        parser.add_argument('--alpha', type=float, default=6.)
        parser.add_argument('--beta', type=float, default=2.)
       
        parser.add_argument('--alpha_lambda', type=float, default=1e-5)
        parser.add_argument('--tv_lambda', type=float, default=1e-5)
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--learning_rate', type=int, default=1e2 )
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--print_iter', type=int, default=25)
        parser.add_argument('--decay_iter', type=int, default=100)
        parser.add_argument('--decay_factor', type=float, default=1e-1)
        parser.add_argument('--gpu', type=int, nargs='*', default=None)

        args = parser.parse_args()

        gpu = args.gpu
        cuda = True if gpu is not None else False
        use_mult_gpu = isinstance(gpu, list)
        if cuda:
            if use_mult_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
       # print(torch.cuda.device_count(), use_mult_gpu, cuda)
        MSE = np.zeros((1,9))
        #quantization = [4,16,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864,1073741824,4294967296]
        quantization = [2,3,4,5,6,7,8,9,10]
        #quantization = [9,10]
        quant_ratio = (1/32)* np.log2(quantization)
        ty_check = 0
        if(ty_check):
            ty = "stoch"
        else:
            ty = "det"
        
        
        
        
        
        
        
        folder = r'/home/arifh/Documents/Research/Image_Inversion/Experiments-Code/Scalar_Q _diff_variance/Pict'

        
        
        
            #print(source)
            #exit()
        for i  in range(1):
            j=0
            for c in quantization:
                err_m = []
                for img in sorted(os.listdir(folder)):
                    im = os.path.join(folder,img)
                    f = "var0_"+ ty +"_"+str(c)+".pt"
                    #f = "Nocomp.pt" 
                    error,noise_error = invert(filename = f,image= im, network=args.network, layer=args.layer, 
                                            alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
                                            tv_lambda=args.tv_lambda, epochs=args.epochs,
                                            learning_rate=args.learning_rate, momentum=args.momentum, 
                                            print_iter=args.print_iter, decay_iter=args.decay_iter,
                                            decay_factor=args.decay_factor, cuda = cuda,comp = c)
            
                    err_m.append(float(error.detach().numpy()))
                    noise_error.detach().numpy()
                err_m =np.array(err_m)    
                MSE[i,j] = np.mean(err_m) # Doubts
                j+=1
        #print(error)
        #print(noise_error)
        #MSE[i,j] = float(error) # Doubts
       
        
        # invert(image=args.image, layer=args.layer, 
        #         alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
        #         tv_lambda=args.tv_lambda, epochs=args.epochs,
        #         learning_rate=args.learning_rate, momentum=args.momentum, 
        #         print_iter=args.print_iter, decay_iter=args.decay_iter,
        #         decay_factor=args.decay_factor, cuda=cuda)
        
        error_mean = np.mean(MSE,axis=0)
        error_std = np.std(MSE,axis=0)
        fig, ax = plt.subplots()
        fig.canvas.draw()
        labels = quantization
        # labels[1] = 'Testing'
       
        with open('MSE_det.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump([error_mean,error_std], file)
            #pickle.dump(error_std,file)
    
        ax.set_xticklabels(quantization)
    
        ax.plot(quantization,error_mean)
        ax.errorbar(quant_ratio,error_mean,yerr= error_std ,color="blue",elinewidth=0.5,barsabove=True)
       
        plt.xlabel("Quantization_Level")
         #ax.xticks( quantization)
        plt.ylabel("MSE error")
        plt.title("Scalar sparsification with Binomial_noise" )
        print(error_mean)
        print(error_std)
        plt.show()
   except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
