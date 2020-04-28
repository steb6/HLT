import os
import pickle
import numpy


def load_embeddings(embeddings_file, embeddings_dim):
    file = 'embeddings/{}{}.pickle'.format(embeddings_file, embeddings_dim)
    if os.path.exists(file):
        # Pickle file exists
        print("Embeddings pickle found!")
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Embeddings pickle not found!")
        file = 'embeddings/{}{}.txt'.format(embeddings_file, embeddings_dim)
        if not os.path.exists(file):
            raise FileNotFoundError
        # Pickle file does not exist
        print("Indexing {}...".format(file))
        embeddings_dict = {}
        f = open(file, "r", encoding="utf-8")
        for i, line in enumerate(f):
            # Get right values
            values = line.split()
            word = values[0]
            coefs = numpy.asarray(values[-embeddings_dim:], dtype='float32')
            # Check if it is an ascii (it removes useless embeddings)
            try:
                word.encode('ascii')
            except:
                continue
            # Insert word - embeddings in dictionary
            embeddings_dict[word] = coefs
        f.close()
        # Save embeddings_dict as pickle file
        with open(os.path.join('embeddings', '{}{}.pickle'.format(embeddings_file, embeddings_dim)), 'wb') as pick_file:
            pickle.dump(embeddings_dict, pick_file)

        return load_embeddings(embeddings_file, embeddings_dim)
