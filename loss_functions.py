import torch

PI = torch.acos(torch.zeros(1)).item() * 2


def quantization_loss(x):
    return (x * (1 - x)).mean(dim=1).mean()


def angular_similarity(x1, x2):
    cosine_sim = torch.nn.CosineSimilarity()(x1, x2).view(-1, 1)

    return 1 - torch.div(torch.acos(cosine_sim), PI)


def hamming_similarity(h1, h2, code_size):
    return 1 - (torch.abs(h1 - h2).sum(dim=1, keepdims=True) / code_size)


def preservation_loss(h1, x1, h2, x2, code_size):
    return (angular_similarity(x1, x2) - hamming_similarity(h1, h2, code_size=code_size)).pow(2).mean()
