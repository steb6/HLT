import numpy


def matrix_wv_generator(vectors):
    wv_map = {}
    pos = 0
    vocab_size = len(vectors)
    dim = len(vectors['sole'])
    # +1 for zero padding token and +1 for unk
    emb_matrix = numpy.ndarray((vocab_size + 2, dim), dtype='float32')
    for i, (word, vector) in enumerate(vectors.items()):
        if len(vector) > 199:
            pos = i + 1
            wv_map[word] = pos
            emb_matrix[pos] = vector

    # add unknown token
    pos += 1
    wv_map["<unk>"] = pos
    emb_matrix[pos] = numpy.random.uniform(low=-0.05, high=0.05, size=dim)
    return emb_matrix, wv_map
