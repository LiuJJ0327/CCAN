import torch
from torch import nn
from torch.nn import functional as F
from base_ae import BaseAE
from typing import List, TypeVar

Tensor = TypeVar('torch.tensor')

class DSNAE(BaseAE):

    def __init__(self, shared_encoder, decoder, input_dim: int, latent_dim: int, 
                 hidden_dims: List = None, dop: float = 0.1, lambda2: float = 0.1, lambda3: float = 0.1, noise_flag: bool = False, norm_flag: bool = False,
                 **kwargs) -> None:
        super(DSNAE, self).__init__()
        self.latent_dim = latent_dim
        self.noise_flag = noise_flag
        self.dop = dop
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.norm_flag = norm_flag

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.shared_encoder = shared_encoder
        self.decoder = decoder
        modules = []

        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0], bias=True),
                #nn.ReLU(),
                nn.Tanh(),
                nn.Dropout(self.dop)
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    #nn.ReLU(),
                    nn.Tanh(),
                    nn.Dropout(self.dop)
                )
            )
        modules.append(nn.Dropout(self.dop))
        modules.append(nn.Linear(hidden_dims[-1], 1, bias=True))  
        self.private_encoder = nn.Sequential(*modules)

    def p_encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.private_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.private_encoder(input)
        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def s_encode(self, input: Tensor) -> Tensor:
        if self.noise_flag and self.training:
            latent_code = self.shared_encoder(input + torch.randn_like(input, requires_grad=False) * 0.1)
        else:
            latent_code = self.shared_encoder(input)
        if self.norm_flag:
            return F.normalize(latent_code, p=2, dim=1)
        else:
            return latent_code

    def encode(self, input: Tensor) -> Tensor:
        p_latent_code = self.p_encode(input)
        s_latent_code = self.s_encode(input)
        return torch.cat((p_latent_code, s_latent_code), dim=1)

    def non_linear(self, z: Tensor) -> Tensor:
        p_latent=z[:, :1]
        non_linear_decode = torch.cat((torch.sin(p_latent),torch.cos(p_latent)),dim=1)
        return non_linear_decode

    def decode(self, z: Tensor) -> Tensor:
        p_latent=z[:, :1]
        s_latent=z[:, (1-z.shape[1]):]
        p_decode=self.non_linear(z)       
        latent_code=torch.cat((p_decode, s_latent), dim=1)
        outputs = self.decoder(latent_code)

        return outputs

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)
        out= self.decode(z)
        return [input, out, z]


    def loss_function(self, *args, **kwargs) -> dict:
        input = args[0]
        recons = args[1]
        z = args[2]

        p_z = z[:, :1]
        s_z = z[:, (1-z.shape[1]):]

        recons_loss = F.mse_loss(input, recons)

        s_l2_norm = torch.norm(s_z, p=2, dim=1, keepdim=True).detach()
        s_l2 = s_z.div(s_l2_norm.expand_as(s_z) + 1e-6)

        p_l2_norm = torch.norm(p_z, p=2, dim=1, keepdim=True).detach()
        p_l2 = p_z.div(p_l2_norm.expand_as(p_z) + 1e-6)

        ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2)) 
        #print('recons_loss:', recons_loss)      
        #print('ortho_loss:', ortho_loss)  
        #loss = recons_loss + 1e-5 * ortho_loss
        loss = self.lambda2*recons_loss + self.lambda3*ortho_loss
        return {'loss': loss, 'recons_loss': recons_loss, 'ortho_loss': ortho_loss}
        #return {'loss': recons_loss, 'recons_loss': recons_loss}
