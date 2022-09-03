import time

import torch
import torch.nn.functional as F

def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError

    return x

class RepresentationAdv():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, epsilon, beta, max_iters, _type='linf'):

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.beta = beta
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        
    def get_loss(self, npts, partial_input, coarse_gt, dense_gt, normal, 
        model, loss_d1, random_start=True):
        # partial_input: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        
        if random_start:
            rand_perturb = torch.FloatTensor(partial_input.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.cuda()
            x =partial_input.clone() + rand_perturb
        else:
            x =partial_input.clone()
        
        model.eval() 
        x.requires_grad = True
        _,n,_ = normal.size()

        with torch.enable_grad():

            for _iter in range(self.max_iters):

                model.zero_grad()

                v, y_coarse, y_detail = model(x, npts, True)
                y_coarse = y_coarse.permute(0, 2, 1)
                y_detail = y_detail.permute(0, 2, 1)

                loss = loss_d1(coarse_gt, y_coarse)

                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]

                proj_grads = torch.einsum('bij,bij->bi', (grads.data.permute(0,2,1), normal)).unsqueeze(2).expand(1, n, 3)*normal

                x.data += self.beta * (grads.data - proj_grads.permute(0,2,1))

                x = project(x,partial_input,self.epsilon, self._type)


        return x.detach()