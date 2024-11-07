import gensim.downloader as api
import numpy as np
import torch


def get_glove_embeddings(vocab: dict[str, int]) -> (torch.Tensor, int):
    """
    Load glove-wiki-gigaword-100 embedding, which are in the vocab.
    :param vocab: vocab with words
    :return: torch tensor embeddings and count of glove words
    """
    word_vectors = api.load("glove-wiki-gigaword-100")

    embeddings = []
    embedding_size = 100
    embeddings.append(np.zeros(embedding_size))

    glove_word_count = 0
    for word in vocab:
        if word_vectors.has_index_for(word):
            embeddings.append(word_vectors[word])
            glove_word_count += 1
        else:
            embeddings.append(np.random.uniform(-1 / np.sqrt(embedding_size),
                                                1 / np.sqrt(embedding_size),
                                                embedding_size))

    embeddings = torch.from_numpy(np.array(embeddings, dtype=float))
    embeddings = embeddings.to(dtype=torch.float32)

    return embeddings, glove_word_count
