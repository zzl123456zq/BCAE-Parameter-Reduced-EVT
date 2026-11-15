import torch
import torch.nn.functional as F
import torch.nn as nn

class FlexibleConv1dAutoencoder(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_size, latent_dim, activation, use_batchnorm=False):
        super(FlexibleConv1dAutoencoder, self).__init__()
        activation_map = {
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(),
            "tanh": nn.Tanh()
        }
        act = activation_map[activation.lower()]
        
        encoder_layers = []
        prev = in_channels
        for ch in conv_channels:
            encoder_layers.append(nn.Conv1d(prev, ch, kernel_size=kernel_size, padding=kernel_size//2))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(ch))
            encoder_layers.append(act)
            prev = ch
        encoder_layers.append(nn.Conv1d(prev, latent_dim, kernel_size=kernel_size, padding=kernel_size//2))
        encoder_layers.append(act)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev = latent_dim
        for ch in reversed(conv_channels):
            decoder_layers.append(nn.ConvTranspose1d(prev, ch, kernel_size=kernel_size, padding=kernel_size//2))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(ch))
            decoder_layers.append(act)
            prev = ch
        decoder_layers.append(nn.ConvTranspose1d(prev, in_channels, kernel_size=kernel_size, padding=kernel_size//2))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    


def contrastive_loss(x, x_recon, tau=0.1, alpha=1, beta=1):
    """Compute the bidirectional input--reconstruction contrastive loss.

    The tensors may contain arbitrary non-batch dimensions; each sample is
    flattened and L2-normalized before its cosine similarities are calculated.
    """

    batch_size = x.size(0)

    # Unit normalization turns the dot products into cosine similarities.
    x = F.normalize(x.view(batch_size,-1), dim=1)
    x_recon = F.normalize(x_recon.view(batch_size,-1), dim=1)

    # Pairwise similarity matrices.
    sim_x_xrecon = torch.matmul(x, x_recon.T)  # (B, B)
    sim_x_x = torch.matmul(x, x.T)             # (B, B)
    sim_xrecon_x = torch.matmul(x_recon, x.T)   # (B, B)
    sim_xrecon_xrecon = torch.matmul(x_recon, x_recon.T) # (B, B)

    # Diagonal input--reconstruction similarities are the positive pairs.
    pos_sim_x = torch.diag(sim_x_xrecon)  # (B,)
    pos_sim_xrecon = torch.diag(sim_xrecon_x)  # (B,)

    # Numerators: exponentiated positive-pair similarities.
    numerator_IC = torch.exp(pos_sim_x / tau)  # (B,)
    numerator_RC = torch.exp(pos_sim_xrecon / tau)  # (B,)

    # Input-anchor denominator: all cross-view pairs and non-self inputs.
    exp_sim_x_xrecon = torch.exp(sim_x_xrecon / tau)  # (B,B)
    exp_sim_x_x = torch.exp(sim_x_x / tau)  # (B,B)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=x.device)
    denom_IC = exp_sim_x_xrecon.sum(dim=1) + exp_sim_x_x.masked_select(mask).reshape(batch_size, -1).sum(dim=1)

    # Reconstruction-anchor denominator with the symmetric role reversal.
    exp_sim_xrecon_x = torch.exp(sim_xrecon_x / tau)
    exp_sim_xrecon_xrecon = torch.exp(sim_xrecon_xrecon / tau)
    denom_RC = exp_sim_xrecon_x.sum(dim=1) + exp_sim_xrecon_xrecon.masked_select(mask).reshape(batch_size, -1).sum(dim=1)

    # Average the two directional losses.
    l_IC = -torch.log(numerator_IC / denom_IC)
    l_RC = -torch.log(numerator_RC / denom_RC)

    loss = (alpha*l_IC + beta*l_RC).mean() / 2

    return loss 
