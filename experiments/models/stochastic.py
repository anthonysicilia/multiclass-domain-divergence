import torch
import torch.nn.functional as F

from math import log, expm1

from ..bounds.functional import dziugaite_variational_bound
from .learners import PyTorchHypothesisSpace

class Gaussian(torch.nn.Module):
    """
    Implementation of a Gaussian random variable, using softplus 
    for the standard deviation and with implementation of sampling 
    and KL divergence computation.
    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.
    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)
    device : string
        Device the code will run in (e.g. 'cpu')
    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be 
        fixed or learnt.
    
    Credit:
    This class contains code adapted from the github repo: 
    https://github.com/mperezortiz/PBB
    with associated paper: 
    https://arxiv.org/abs/2007.12911
    """

    def __init__(self, mu, rho, device='cpu', fixed=False):
        super().__init__()
        self.mu = torch.nn.Parameter(mu.to(device), requires_grad=not fixed)
        self.rho = torch.nn.Parameter(rho.to(device), requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always 
        # positive during the optimisation. Specifically, we use 
        # sigma = log(exp(rho)+1)

        # Correction per issue: 
        # https://github.com/mperezortiz/PBB/issues/1
        # OLD CODE:
        # m = nn.Softplus()
        # return m(self.rho)
        # NEW CODE:
        return torch.log(1 + torch.exp(self.rho))

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and 
        # other, refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div

class ProbLinear(torch.nn.Module):
    """
    Implementation of a Probabilistic Linear layer.
    Parameters
    ----------
    init_layer : Linear object
        Linear layer object used to initialize
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cpu')

    Credit:
    This class contains code adapted from the github repo: 
    https://github.com/mperezortiz/PBB
    with associated paper: 
    https://arxiv.org/abs/2007.12911
    """

    def __init__(self, init_layer, rho_prior, dist='gaussian', 
        device='cpu'):
        super().__init__()

        self.device = device

        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias
        weights_rho_init = torch.ones_like(weights_mu_init) \
            * rho_prior
        bias_rho_init = torch.ones_like(bias_mu_init) * rho_prior

        if dist == 'gaussian':
            dist = Gaussian
        else:
            raise RuntimeError(f'Prior dist not supported: {dist}')
        
        self.dist_init = dist
        self.bias = dist(bias_mu_init.clone(), 
            bias_rho_init.clone(), 
            device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(), 
            weights_rho_init.clone(), 
            device=device, fixed=False)
        self.weight_prior = dist(weights_mu_init.clone(), 
            weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(bias_mu_init.clone(), 
            bias_rho_init.clone(), 
            device=device, fixed=True)

        self.kl_div = 0
        self.init_current_weights()
    
    def sample(self):
        self.current_weight = self.weight.sample()
        self.current_bias = self.bias.sample()
    
    def mean(self):
        self.current_weight = self.weight.mu
        self.current_bias = self.bias.mu
    
    def init_current_weights(self):
        # default to None. Promotes proper use and prevents 
        # gradients from hanging around...
        self.current_weight = None
        self.current_bias = None
        self.current_weight = None
        self.current_bias = None

    def forward(self, input):
        weight = self.current_weight
        bias = self.current_bias
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) \
                + self.bias.compute_kl(self.bias_prior)
        return F.linear(input, weight, bias)

class ProbConv2d(torch.nn.Module):
    """
    Implementation of a Probabilistic Convolutional layer.
    Parameters
    ----------
    init_layer : Conv2d object
        Conv2d layer object used to initialize
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cpu')
    
    Credit:
    This class contains code adapted from the github repo: 
    https://github.com/mperezortiz/PBB
    with associated paper: 
    https://arxiv.org/abs/2007.12911
    """

    def __init__(self, init_layer, rho_prior, dist='gaussian', 
        device='cpu'):
        super().__init__()

        self.stride = init_layer.stride
        self.padding = init_layer.padding
        self.dilation = init_layer.dilation
        self.groups = init_layer.groups
        self.device = device

        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias

        # set scale parameters
        weights_rho_init = torch.ones_like(weights_mu_init) \
            * rho_prior
        if bias_mu_init is not None:
            bias_rho_init = torch.ones_like(bias_mu_init) \
                * rho_prior

        if dist == 'gaussian':
            dist = Gaussian
        else:
            raise RuntimeError(f'Wrong prior_dist {dist}')

        self.weight = dist(weights_mu_init.clone(), weights_rho_init.clone(), 
            device=device, fixed=False)
        if bias_mu_init is not None:
            self.bias = dist(bias_mu_init.clone(), 
                bias_rho_init.clone(),
                device=device, fixed=False)
        else:
            self.bias = None
        self.weight_prior = dist(weights_mu_init.clone(), 
            weights_rho_init.clone(), device=device, fixed=True)
        if bias_mu_init is not None:
            self.bias_prior = dist(bias_mu_init.clone(), 
                bias_rho_init.clone(),
                device=device, fixed=True)

        self.kl_div = 0
        self.init_current_weights()
    
    def sample(self):
        self.current_weight = self.weight.sample()
        self.current_bias = self.bias.sample()
    
    def mean(self):
        self.current_weight = self.weight.mu
        self.current_bias = self.bias.mu
    
    def init_current_weights(self):
        # default to None. Promotes proper use and prevents 
        # gradients from hanging around after init
        self.current_weight = None
        self.current_bias = None
        self.current_weight = None
        self.current_bias = None

    def forward(self, input):
        weight = self.current_weight
        bias = self.current_bias
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) \
                + (self.bias.compute_kl(self.bias_prior) 
                    if self.bias is not None else 0)
        return F.conv2d(input, weight, bias, self.stride, self.
            padding, self.dilation, self.groups)

class ProbConvTranspose2d(torch.nn.Module):
    """
    Implementation of a Probabilistic Deconvolutional layer.
    Parameters
    ----------
    init_layer : ConvTranspose2d object
        ConvTranspose2d layer object used to initialize
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)
    dist : string
        string that indicates the type of distribution for the
        prior and posterior
    device : string
        Device the code will run in (e.g. 'cpu')
    """
    def __init__(self, init_layer, rho_prior, dist='gaussian', 
        device='cpu'):
        super().__init__()

        self.stride = init_layer.stride
        self.padding = init_layer.padding
        self.output_padding = init_layer.output_padding
        self.groups = init_layer.groups
        self.dilation = init_layer.dilation
        self.device = device

        weights_mu_init = init_layer.weight
        bias_mu_init = init_layer.bias

        # set scale parameters, NOTE: this is transposed
        weights_rho_init = torch.ones_like(weights_mu_init) \
            * rho_prior
        if bias_mu_init is not None:
            bias_rho_init = torch.ones_like(bias_mu_init) \
                * rho_prior

        if dist == 'gaussian':
            dist = Gaussian
        else:
            raise RuntimeError(f'Wrong prior_dist {dist}')

        self.weight = dist(weights_mu_init.clone(), 
            weights_rho_init.clone(),
            device=device, fixed=False)
        if bias_mu_init is not None:
            self.bias = dist(bias_mu_init.clone(), 
                bias_rho_init.clone(),
                device=device, fixed=False)
        self.weight_prior = dist(weights_mu_init.clone(), 
            weights_rho_init.clone(), device=device, fixed=True)
        if bias_mu_init is not None:
            self.bias_prior = dist(bias_mu_init.clone(), 
                bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0
        self.init_current_weights()
    
    def sample(self):
        self.current_weight = self.weight.sample()
        self.current_bias = self.bias.sample()
    
    def mean(self):
        self.current_weight = self.weight.mu
        self.current_bias = self.bias.mu
    
    def init_current_weights(self):
        # default to None. Promotes proper use and prevents 
        # gradients from hanging around...
        self.current_weight = None
        self.current_bias = None
        self.current_weight = None
        self.current_bias = None

    def forward(self, input):
        weight = self.current_weight
        bias = self.current_bias
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) \
                + (self.bias.compute_kl(self.bias_prior) 
                    if self.bias is not None else 0)
        return F.conv_transpose2d(input, weight, bias, self.stride,
            self.padding, self.output_padding, self.groups, 
            self.dilation)

class Stochastic(PyTorchHypothesisSpace):

    CONVERSION_MAP = {
        torch.nn.Linear : ProbLinear,
        torch.nn.Conv2d : ProbConv2d,
        torch.nn.ConvTranspose2d : ProbConvTranspose2d
    }

    def __init__(self, hypothesis_space, prior=None, m=1000, delta=0.05,
        sigma_prior=0.01, device='cpu'):
        super().__init__(
            hypothesis_space.num_classes,
            hypothesis_space.epochs,
            hypothesis_space.optimizer,
            hypothesis_space.scheduler,
            hypothesis_space.batch_size,
            hypothesis_space.finetune)
        self.underlying = hypothesis_space
        self.m = m
        self.delta = delta
        self.prior = prior
        self.device = device
        self.rho_prior = log(expm1(sigma_prior))
    
    def _make_stochastic(self, module, rho_prior, dist='gaussian', 
        device='cuda'):

        immediate_frozen_children = list(module.children())
        if type(module) == torch.nn.Sequential:
            for i in range(len(module)):
                for k,v in self.CONVERSION_MAP.items():
                    if type(module[i]) == k:
                        module[i] = v(module[i], rho_prior, 
                            dist=dist, device=device)
        else:
            for attr_str in dir(module):
                target_attr = getattr(module, attr_str)
                new_mod = None
                for k,v in self.CONVERSION_MAP.items():
                    if type(target_attr) == k:
                        new_mod = v(target_attr, rho_prior, 
                            dist=dist, device=device)
                if new_mod is not None:
                    setattr(module, attr_str, new_mod)
        for child in immediate_frozen_children:
            self._make_stochastic(child, rho_prior, dist=dist, device=device)
    
    def __call__(self):
        h = self.underlying()
        if self.prior is not None:
            h.load_state_dict(self.prior.state_dict())
        self._make_stochastic(h, self.rho_prior)
        return h
    
    def loss_fn(self, yhat, y, w=None, h=None):
        loss = super().loss_fn(yhat, y, w=w)
        if h is not None:
            # infer if we should regularize based on h
            kl = compute_kl(h, self.device)
            eps = dziugaite_variational_bound(loss, kl, 
                self.m, self.delta)
            return loss + eps
        else:
            return loss

def freeze_batchnorm(model):
    """
    From the following PyTorch discussion:
    "https://discuss.pytorch.org/t/
    how-to-freeze-bn-layers-while-training-the
    -rest-of-network-mean-and-var-wont-freeze/89736/11"
    """
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad_(False)
            module.eval()

def isprob(module):
    check = False
    for layer_type in Stochastic.CONVERSION_MAP.values():
        check = check or isinstance(module, layer_type)
    return check

def compute_kl(model, device):
    kl_div = torch.zeros(1, requires_grad=True).to(device)
    for m in model.modules():
        # find top level prob modules and sum.
        # multivariate normal with dialog cov
        # is a product distr of i.i.d uni normals
        # so we can just sum the kl divergences
        if isprob(m): kl_div = kl_div + m.kl_div
    return kl_div

def sample(model):
    for m in model.modules():
        if isprob(m): m.sample()
    return model

def mean(model):
    for m in model.modules():
        if isprob(m): m.mean()
    return model