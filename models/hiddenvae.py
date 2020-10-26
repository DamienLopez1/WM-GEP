"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        '''
        self.fc1 = nn.Linear(latent_size, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 64, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, img_channels, 1, stride=2)
        '''
        self.fc1 = nn.Linear(latent_size,64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,256)
        
        

    def forward(self, x): # pylint: disable=arguments-differ
        #print('decoder input', x.shape)
        x = F.relu(self.fc1(x))
        #print('decoder fc1 output', x.shape)
        
        
        x = F.relu(self.fc2(x))
        #print('decoder fc2 output', x.shape)
        reconstruction = F.relu(self.fc3(x))
        #print('recon shape', reconstruction.shape)
        #reconstruction = reconstruction.reshape(-1,7,7,3)
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ VAE encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        #self.img_size = img_size
        self.img_channels = img_channels
         
        
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,64)
        
        
        
        self.fc_mu = nn.Linear(64, latent_size)
        self.fc_logsigma = nn.Linear(64, latent_size)


    def forward(self, x): # pylint: disable=arguments-differ
        #print('encode input', x.shape)
        x = F.relu(self.fc1(x))
        #print('Encoder fc1 output',x.shape)
        x = F.relu(self.fc2(x))
        #print('Encoder fc2 output',x.shape)
        #x = x.view(x.size(0), -1)
        #print('view shape', x.shape)
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)

        return mu, logsigma

class HiddenVAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(HiddenVAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
