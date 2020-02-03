import numpy as np

ALPHABET = {c: i for i, c in enumerate('Pabcdefghijklmnopqrstuvwxyz')}
PAD = 'P'


def word2idx(word, max_len, dtype=np.uint8):
    word = word + PAD * (max_len-len(word))
    return np.array(list(map(lambda c: ALPHABET[c], word)), dtype=dtype)
