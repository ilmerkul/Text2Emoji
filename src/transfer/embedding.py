import gensim.downloader as api
import numpy as np
import torch


def get_glove_embbedings(vocab):
    word_vectors = api.load("glove-wiki-gigaword-100")

    embbedings = []
    embbeding_size = 100
    # pad
    embbedings.append(np.zeros(embbeding_size))

    glove_word_count = 0
    for word in vocab:
        if word_vectors.has_index_for(word):
            embbedings.append(word_vectors[word])
            glove_word_count += 1
        else:
            embbedings.append(
                np.random.uniform(-1 / np.sqrt(embbeding_size), 1 / np.sqrt(embbeding_size), embbeding_size))

    print(f'glove_word_count: {glove_word_count}, size of vocab: {len(vocab)}')

    embbedings = torch.from_numpy(np.array(embbedings, dtype=float)).to(dtype=torch.float32)

    return embbedings, embbeding_size
