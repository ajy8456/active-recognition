import torch
from torch import nn
from torch.distributions import Normal
from modeling.classifier_models import Encoder, Decoder

class Classifier_NP(nn.Module):
    """
       Simple Image classifier with CNP(Conditional Neural Process)
       Returns:
           p_y_pred: Normal distribution of Mu, sigma (Y target)
    """
    def __init__(self, x_dim, y_dim, r_dim, h_dim):
        super(Classifier_NP, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.h_dim = h_dim

        # Initialize networks
        self.xy_to_r = Encoder(x_dim, y_dim, h_dim, r_dim)
        self.xr_to_y = Decoder(x_dim, r_dim, h_dim, y_dim)

    def aggregate(self, r_i):
        return torch.mean(r_i, dim=1)

    def xy_to_representation(self, x, y):
        batch_size, num_points, _ = x.size()
        x_flat = x.view(batch_size * num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size * num_points, self.y_dim)
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        r = self.aggregate(r_i)
        return r

    def forward(self, x_context, y_context, x_target, y_target=None):
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()
        r = self.xy_to_representation(x_target, y_target)
        y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, r)
        p_y_pred = Normal(y_pred_mu, y_pred_sigma)
        return p_y_pred