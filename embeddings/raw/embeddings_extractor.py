from gensim.models import Word2Vec
import tqdm

input_emb = "rawwiki_iter=5_algorithm=skipgram_window=10_size=300_neg-samples=10.m"

output_emb = "../embeddings300.txt"

model = Word2Vec.load(input_emb)

with open(output_emb, "w+") as f:
    for word in tqdm.tqdm(list(model.wv.vocab.keys())):
        if len(word) > 0:
            f.write(str(word)+' ')
            for elem in model.wv[word]:
                f.write(str(elem)+' ')
            f.write('\n')
