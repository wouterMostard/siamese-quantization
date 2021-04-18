import argparse
import pickle
import logging
from datetime import date

import torch
from torch.utils.data import DataLoader

from models.SiameseAutoencoder import SiameseAutoencoder
from loss_functions import quantization_loss, preservation_loss
from helpers import load_vectors, SiameseDataset

EMBEDDING_SIZE = 300
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit-size", default=256, help="Bit size for the latent layer")
    parser.add_argument("--lr", default=0.0001, help="Learning rate for the auto-encoders")
    parser.add_argument("--method", default='siamese', help="Method to learn", choices=["siamese", "auto", "lsh"])
    parser.add_argument("--embedding-path",
                        default="./data/embeddings/crawl-300d-2M.vec",
                        help="Path to embeddings, defaults to fasttext")
    parser.add_argument("--maxload", help="Maximum load for embeddings", default=100000)
    parser.add_argument("--batch-size", default=128, help="Batch size")
    parser.add_argument("--nns-path", default="./data/neighbors/nns100k.pkl")
    parser.add_argument("--n-epochs", default=100)
    parser.add_argument("--val-size", default=500)
    parser.add_argument("--modelname", default=None)

    args = parser.parse_args()

    train(args)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_validation = float("inf")

    if not args.modelname:
        model_name = date.today().strftime("%d-%m-%Y")
    else:
        model_name = args.modelname

    logging.info(f"Training on device: {device}")
    recon_loss = torch.nn.MSELoss(reduction='mean')

    logging.info("Loading data")
    embeddings, _ = load_vectors(args.embedding_path, max_number_words=int(args.maxload))
    logging.info(f"{embeddings.shape[0]} embeddings loaded")

    with open(args.nns_path, 'rb') as file:
        nns = pickle.load(file)
    logging.info(f"{len(nns)} neighbors loaded")

    validation_set = torch.from_numpy(embeddings[-args.val_size:]).float()

    dataset = SiameseDataset(embeddings=torch.from_numpy(embeddings[:-args.val_size]).float().to(device), nns=nns)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SiameseAutoencoder(EMBEDDING_SIZE, args.bit_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        model.train()

        for batch in dataloader:
            x1 = batch[0]
            x2 = batch[1]

            optimizer.zero_grad()

            r1, h1, b1, r2, h2, b2 = model(x1, x2)

            rec_loss = (recon_loss(r1, x1) + recon_loss(r2, x2)) / 2  # Reconstruction loss
            qua_loss = (quantization_loss(h1) + quantization_loss(h2)) / 2  # Quantization loss
            pre_loss = preservation_loss(b1, x1, b2, x2, code_size=args.bit_size)  # Preserving loss

            total_loss = rec_loss + qua_loss + pre_loss
            total_loss.backward()

            optimizer.step()

        with torch.no_grad():
            model.eval()

            rv1, hv1, bv1, rv2, hv2, bv2 = model(validation_set, validation_set)

            rec_loss = (recon_loss(rv1, validation_set) + recon_loss(rv2, validation_set)) / 2
            qua_loss = (quantization_loss(hv1) + quantization_loss(hv2)) / 2
            pre_loss = preservation_loss(bv1, validation_set, bv2, validation_set, code_size=args.bit_size)

            val_loss = rec_loss + qua_loss + pre_loss

            if val_loss < best_validation:
                logging.info(f"[{epoch + 1}/{args.n_epochs}] Better validation loss found: {val_loss.round(5)}, saving model")
                best_validation = val_loss

                torch.save(model, f'./data/models/{model_name}')


if __name__ == "__main__":
    main()