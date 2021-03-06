import argparse
import pickle
import logging
from datetime import date

import torch
import numpy as np
from torch.utils.data import DataLoader

from models.SiameseAutoencoder import SiameseAutoencoder
from loss_functions import quantization_loss, preservation_loss
from helpers import load_vectors, SiameseDataset, load_20ng_data, preprocess_20ng_data

EMBEDDING_SIZE = 300
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bit-size", default=256, help="Bit size for the latent layer", type=int)
    parser.add_argument("--lr", default=0.001, help="Learning rate for the auto-encoders", type=float)
    parser.add_argument("--method", default='siamese', help="Method to learn", choices=["siamese", "auto", "lsh"], type=str)
    parser.add_argument("--embedding-path",
                        default="./data/embeddings/crawl-300d-2M.vec",
                        help="Path to embeddings, defaults to fasttext", type=str)
    parser.add_argument("--maxload", help="Maximum load for embeddings", default=100000, type=int)
    parser.add_argument("--batch-size", default=128, help="Batch size", type=str)
    parser.add_argument("--nns-path", default="./data/neighbors/nns100k.pkl", type=str)
    parser.add_argument("--n-epochs", default=100, type=int)
    parser.add_argument("--val-size", default=500, type=int)
    parser.add_argument("--modelname", default=None)
    parser.add_argument("--use-nn", default=False, type=bool)
    parser.add_argument("--pre-train", default=100, type=int)

    args = parser.parse_args()

    train(args)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_validation = 0.0
    num_not_improved = 0

    if not args.modelname:
        model_name = date.today().strftime("%d-%m-%Y")
    else:
        model_name = args.modelname

    logging.info(f"Training on device: {device}")
    recon_loss = torch.nn.MSELoss(reduction='mean')

    logging.info("Loading data")
    embeddings, words = load_vectors(args.embedding_path, max_number_words=int(args.maxload))
    idx2word = {idx: word for idx, word in enumerate(words)}
    word2idx = {word: idx for idx, word in idx2word.items()}

    data = load_20ng_data('./data/evaluation//20ng-train-no-stop.txt').sample(frac=1).reset_index()
    data_test = load_20ng_data('./data/evaluation/20ng-test-no-stop.txt').sample(frac=0.1).reset_index()

    input_ = preprocess_20ng_data(data, normalize=False, embeddings=embeddings, word2idx=word2idx)
    input_ = torch.from_numpy(input_).float().to(device)
    input_test = preprocess_20ng_data(data_test, normalize=False, embeddings=embeddings, word2idx=word2idx)
    input_test = torch.from_numpy(input_test).float().to(device)

    logging.info(f"{embeddings.shape[0]} embeddings loaded")

    with open(args.nns_path, 'rb') as file:
        nns = pickle.load(file)
    logging.info(f"{len(nns)} neighbors loaded")

    dataset = SiameseDataset(embeddings=torch.from_numpy(embeddings[:-args.val_size]).float().to(device), nns=nns)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SiameseAutoencoder(EMBEDDING_SIZE, args.bit_size).to(device)
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

            if epoch > args.pre_train:
                total_loss = rec_loss + qua_loss + pre_loss
            else:
                total_loss = rec_loss + pre_loss

            total_loss.backward()

            optimizer.step()

        with torch.no_grad():

            model.eval()

            hidden, binary = model.get_binary_codes(input_)
            binary_input = binary.detach().cpu().numpy().astype(np.uint8)
            hidden, binary = model.get_binary_codes(input_test)
            binary_test = binary.detach().cpu().numpy().astype(np.uint8)

            precisions = []

            for idx, row in enumerate(binary_test):
                distances = (row ^ binary_input).sum(axis=1)
                precisions.append(
                    (data.iloc[np.argsort(distances)[:100]]['label'] == data_test.iloc[idx]['label']).sum() / 100)

            val_prec = np.mean(precisions)

            if val_prec > best_validation:
                num_not_improved = 0
                logging.info(f"[{epoch + 1}/{args.n_epochs}] Better validation loss found: "
                             f"P@100loss: {round(val_prec.item(), 4)} \t saving model")

                best_validation = val_prec
                torch.save(model, f'./data/models/{model_name}')
            else:
                num_not_improved += 1

                if num_not_improved == 50:
                    logging.info("Not increased for 50 epochs, stopping")
                    exit()


if __name__ == "__main__":
    main()
