import gensim.downloader as api
import numpy as np
import torch


def get_glove_embbedings(vocab: dict[str, int]) -> (torch.Tensor, int):
    """
    Load glove-wiki-gigaword-100 embbeding, which are in the vocab.
    :param vocab: vocab with words
    :return: torch tensor embbedings and count of glove words
    """
    word_vectors = api.load("glove-wiki-gigaword-100")

    embbedings = []
    embbeding_size = 100
    embbedings.append(np.zeros(embbeding_size))

    glove_word_count = 0
    for word in vocab:
        if word_vectors.has_index_for(word):
            embbedings.append(word_vectors[word])
            glove_word_count += 1
        else:
            embbedings.append(
                np.random.uniform(-1 / np.sqrt(embbeding_size), 1 / np.sqrt(embbeding_size), embbeding_size))

    embbedings = torch.from_numpy(np.array(embbedings, dtype=float)).to(dtype=torch.float32)

    return embbedings, glove_word_count
