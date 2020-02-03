import numpy as np

ALPHABET = {c: i for i, c in enumerate('Pabcdefghijklmnopqrstuvwxyz')}


def word2idx(word, dtype=np.uint8):
    return np.array(list(map(lambda c: ALPHABET[c], word)), dtype=dtype)
