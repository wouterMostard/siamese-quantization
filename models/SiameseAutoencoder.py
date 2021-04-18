import torch.nn as nn


class SiameseAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(SiameseAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.Sigmoid()
        )

        self.decoder = nn.Linear(latent_size, input_size, bias=False)

    def get_binary_codes(self, x):
        encoded = self.encoder(x)
        binary = (encoded > 0.50).float()
        binary = encoded + (binary - encoded).detach()

        return encoded, binary

    def forward(self, x1, x2):
        encoded1, binary1 = self.get_binary_codes(x1)
        encoded2, binary2 = self.get_binary_codes(x2)

        return self.decoder(binary1), encoded1, binary1, self.decoder(binary2), encoded2, binary2
