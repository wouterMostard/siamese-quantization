from typing import List

from torch.utils.data import Dataset
import numpy as np
import pandas as pd


def load_vectors(embedding_filename: str, max_number_words: int = 2000000):
    with open(embedding_filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as embedding_file:
        next(embedding_file)
        data: List[np.ndarray] = []
        words: List[str] = []
        counter: int = 0
        line: str

        for line in embedding_file:
            if counter == max_number_words:
                break

            tokens: List[str] = line.rstrip().split(' ')
            counter += 1
            embedding: np.ndarray = np.array(list(map(float, tokens[1:])))

            data.append(embedding)
            words.append(tokens[0])

    x = np.vstack(data)

    return x, words


class SiameseDataset(Dataset):
    def __init__(self, embeddings, nns):
        self.embeddings = embeddings
        self.nns = nns

    def __getitem__(self, idx):
        x1 = self.embeddings[idx, :]

        if idx in self.nns:
            random_other = np.random.choice(self.nns[idx])
        else:
            random_other = np.random.randint(0, self.embeddings.shape[0])

        x2 = self.embeddings[random_other, :]

        return x1, x2

    def __len__(self):
        return self.embeddings.shape[0]


def preprocess_20ng_data(data, word2idx, embeddings, normalize=True):
    reps = []

    for idx, row in data.iterrows():
        indices = [word2idx[word] if word in word2idx else 0 for word in row['text'][:1000].split()]

        if len(indices) == 0:
            reps.append(np.zeros(300))
        else:
            reps.append(np.sum([embeddings[index] for index in indices], axis=0))

    input_data = np.vstack(reps)

    if normalize:
        input_data = input_data / (np.linalg.norm(input_data, axis=1, keepdims=True) + 1e-20)

    return input_data


def load_20ng_data(path):
    data = pd.read_csv(path, sep='\t', header=None)
    data.columns = ['label', 'text']

    return data
