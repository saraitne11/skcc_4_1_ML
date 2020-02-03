import numpy as np

def idx2chr(index):
    return chr(ord('a') + index - 1)

def idx2word(indices):
    word = ''
    for char in indices:
        word += idx2chr(char)

    return word

def expectation_to_compound(word, location):
    word = np.array(word)
    length = np.where(word == 0)[0]
    if len(length) == 0:
        length = len(word)
    else:
        length = length[0]
    compound = idx2word(word[:length])
    compounds = []

    location = np.array(location)
    indices = np.where(location == 1)[0]
    idx = 0
    for index in indices[1:]:
        compounds.append(compound[idx:index])
        idx = index
    compounds.append(compound[idx:length])

    return [compound, compounds]