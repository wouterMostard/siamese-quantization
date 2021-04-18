from typing import List

from torch.utils.data import Dataset
import numpy as np


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
