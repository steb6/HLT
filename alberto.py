from transformers import AutoTokenizer, TFAutoModel
import numpy as np
from utilities.dataset_loader import load_dataset
from kutilities.helpers.data_preparation import labels_to_categories, categories_to_onehot, get_labels_to_categories_map
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = TFAutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
tok = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

# Parameters
data_file = 'data/train.tsv'
test_file = 'data/test.tsv'

# Load dataset
training = load_dataset(data_file)
testing = load_dataset(test_file)

# Remove ids converting dict into list
training = list(training.values())

# Get x and y from training list
x = [elem[1] for elem in training]
y = [elem[0] for elem in training]

# test
res_cane = model.predict(np.array([tok.vocab["cane"]]))
res_gatto = model.predict(np.array([tok.vocab["gatto"]]))
res_ruota = model.predict(np.array([tok.vocab["ruota"]]))
print(cosine_similarity(res_cane[1], res_gatto[1]))
print(cosine_similarity(res_cane[1], res_ruota[1]))

# Process x
x_processed = [] # ritorna lista di tuple
for topic, review in x:
    ind_topic = [tok.vocab[token] for token in tok.tokenize(topic)]
    ind_review = [tok.vocab[token] for token in tok.tokenize(review)]
    t = (ind_topic, ind_review)
    x_processed.append(t)

# Process y
y_train_categories = labels_to_categories(y)
y_train_one_hot = categories_to_onehot(y_train_categories)
y_train_map = get_labels_to_categories_map(y)
