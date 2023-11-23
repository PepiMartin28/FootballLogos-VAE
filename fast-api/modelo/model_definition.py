import torch;
import torch.nn as nn
import torch.nn.functional as F

n_G = 32

class AUG_block(nn.Module):
    def __init__(self, out_channels, in_channels=4, kernel_size=5, strides=2,
                 padding=1, **kwargs):
        super(AUG_block, self).__init__(**kwargs)
        self.conv2d_trans = nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size, strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d_trans(X)))

class DEC_block(nn.Module):
    def __init__(self, out_channels, in_channels=4, kernel_size=5, strides=2,
                padding=1, alpha=0.2, **kwargs):
        super(DEC_block, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size,
                                strides, padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(alpha, inplace=True)

    def forward(self, X):
        return self.activation(self.batch_norm(self.conv2d(X)))
    
class Variational_Encoder(nn.Module):
    def __init__(self, latent_dims, n_channels=4):
        super(Variational_Encoder, self).__init__()
        self.conv_seq = nn.Sequential(
            DEC_block(in_channels=n_channels, out_channels=n_G),
            DEC_block(in_channels=n_G, out_channels=n_G*2),
            DEC_block(in_channels=n_G*2, out_channels=n_G*4),
            DEC_block(in_channels=n_G*4, out_channels=n_G*8),
            DEC_block(in_channels=n_G*8, out_channels=n_G*16),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(latent_dims)
        )
        self.linear3 = nn.LazyLinear(latent_dims)
        self.linear4 = nn.LazyLinear(latent_dims)

    def forward(self, x):
        z = self.conv_seq(x)
        media = self.linear3(z)
        log_var = F.relu(self.linear4(z))
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        latente = eps.mul(std).add_(media)
        return (latente, media, log_var)
    
class Decoder(nn.Module):
    def __init__(self, latent_dims, n_channels=4):
        super(Decoder, self).__init__()
        self.seq = nn.Sequential(
            AUG_block(in_channels=latent_dims, out_channels=n_G*16, strides=1, padding=0), 
            AUG_block(in_channels=n_G*16, out_channels=n_G*8), 
            AUG_block(in_channels=n_G*8, out_channels=n_G*4), 
            AUG_block(in_channels=n_G*4, out_channels=n_G*2), 
            AUG_block(in_channels=n_G*2, out_channels=n_G, strides=3, padding = 9), 
            nn.ConvTranspose2d(in_channels=n_G, out_channels=4, kernel_size=2, stride=2, padding=25, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.seq(z)

class Variational_Autoencoder(nn.Module):
    def __init__(self, latent_dims, n_channels=4):
        super(Variational_Autoencoder, self).__init__()
        self.encoder = Variational_Encoder(latent_dims, n_channels)
        self.decoder = Decoder(latent_dims, n_channels)

    def forward(self, x):
        z, media, log_var = self.encoder(x)
        z = z.unsqueeze(2).unsqueeze(3)
        return self.decoder(z), media, log_var