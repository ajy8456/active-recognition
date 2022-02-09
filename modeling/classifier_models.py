import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    """
       Encoder of Classifier_NP(image classifier)
       Returns:
           representation of each train dataset(X,Y)
    """
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        layers = [nn.Linear(x_dim + y_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, r_dim)]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        input_pairs = torch.cat((x, y), dim=1)
        return self.input_to_hidden(input_pairs)

class Decoder(nn.Module):
    """
       Decoder of Classifier_NP(image classifier)
       Returns:
           Mu, sigma of Y target
    """
    def __init__(self, x_dim, r_dim, h_dim, y_dim):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.r_dim = r_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        layers = [nn.Linear(x_dim + r_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.xr_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x, z):
        batch_size, num_points, _ = x.size()
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        x_flat = x.view(batch_size * num_points, self.x_dim)
        z_flat = z.view(batch_size * num_points, self.z_dim)
        input_pairs = torch.cat((x_flat, z_flat), dim=1)
        hidden = self.xr_to_hidden(input_pairs)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma
