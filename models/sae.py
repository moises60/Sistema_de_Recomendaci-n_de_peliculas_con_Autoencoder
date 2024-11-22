import torch
import torch.nn as nn

class SAE(nn.Module):
    """
    Stacked AutoEncoder para sistemas de recomendación.

    Args:
        nb_movies (int): Número de películas.
        layers (list): Lista con el tamaño de las capas ocultas.

    Attributes:
        encoder (nn.Sequential): Capas del encoder.
        decoder (nn.Sequential): Capas del decoder.
    """

    def __init__(self, nb_movies, layers):
        super(SAE, self).__init__()
        self.nb_movies = nb_movies

        # Definir las capas del encoder
        encoder_layers = []
        input_size = nb_movies
        for output_size in layers:
            encoder_layers.append(nn.Linear(input_size, output_size))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(p=0.5))
            input_size = output_size
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        layers.reverse()
        for output_size in layers[1:] + [nb_movies]:
            decoder_layers.append(nn.Linear(input_size, output_size))
            decoder_layers.append(nn.ReLU())
            input_size = output_size
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
