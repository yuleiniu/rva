import torch
from torch import nn
import torch.nn.functional as F

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False

        self.temperature = temperature
        
    def cuda(self):
        self.gpu = True
    
    def cpu(self):
        self.gpu = False
        
    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return noise.detach().cuda()
        else:
            return noise.detach()

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.detach()) # 0.4
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor.detach()
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, -1)
        return soft_samples
    
    def gumbel_softmax(self, logits, temperature):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        if self.training:
            y = self.gumbel_softmax_sample(logits, temperature)
            _, max_value_indexes = y.detach().max(1, keepdim=True)
            y_hard = logits.detach().clone().zero_().scatter_(1, max_value_indexes, 1)
            y = y_hard - y.detach() + y 
        else:
            _, max_value_indexes = logits.detach().max(1, keepdim=True)
            y = logits.detach().clone().zero_().scatter_(1, max_value_indexes, 1)
        return y

    def forward(self, logits, temperature=None):
        samplesize = logits.size()

        if temperature == None:
            temperature = self.temperature

        return self.gumbel_softmax(logits, temperature=temperature)