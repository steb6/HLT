import pickle
from utilities.dataset_loader import load_dataset
from utilities.dataset_indexxer import index_y, index_x
import pandas as pd
from utilities.model import model
import numpy
from utilities.eng_to_ita import traduction
from utilities.y_converter import y_label_to_category_map, y_category_to_onehot, onehot_to_category
from transformers import AutoTokenizer, TFAutoModel
import tqdm
import os
import pickle
import numpy as np

# If we want to save results as Absita file
ABSITA = True

# Test file
test_file = 'data/test.tsv'

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to load the model
best_model = "experiments/alberto/acd/checkpoint"

# Load data
testing = load_dataset(test_file)

# Converting data
ids = testing.keys()
testing = list(testing.values())
x_test_base = [elem[1] for elem in testing]
y_test_base = [elem[0] for elem in testing]
print("Data rode "+str(len(testing))+" lines")
tok = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

# Generate test embedded
if not os.path.isfile("data/alberto/acd/test_embedded.pickle"):

    # Load model
    model = TFAutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

    # Process x


    def index_pad_embed(x, y):
        results = []
        for couple, sent in tqdm.tqdm(zip(x, y), total=len(x)):
            top = couple[0]
            rew = couple[1]
            ind_topic = [tok.vocab[token] for token in tok.tokenize(top)]
            ind_review = [tok.vocab[token] for token in tok.tokenize(rew)]
            while len(ind_review) < text_max_length:
                ind_review.append(0)
            if len(ind_review) > text_max_length:
                ind_review = ind_review[:50]
            emb_topic = model.predict(ind_topic)[1]
            emb_review = model.predict(ind_review)[1]
            r1 = (emb_topic, emb_review)
            r2 = (sent, r1)
            results.append(r2)
        return results


    # Index y: one hot encoding
    y_test_categories = [y_label_to_category_map[elem] for elem in y_test_base]
    y_test_one_hot = [y_category_to_onehot[elem] for elem in y_test_categories]
    y_category_to_label = {v: k for k, v in y_label_to_category_map.items()}
    print("y indexed")

    # Index pad embed
    test_embedded = index_pad_embed(x_test_base, y_test_one_hot)
    test_embedded = [elem[1] for elem in test_embedded]
    pickle.dump(test_embedded, open("data/alberto/acd/test_embedded.pickle", "wb"))
else:
    test_embedded = pickle.load(open("data/alberto/acd/test_embedded.pickle", "rb"))

# We don't need the y values
input1 = np.array([elem[1][1] for elem in test_embedded])
input2 = np.array([elem[1][0] for elem in test_embedded])
test_embedded = [input2, input1]
#print(test_embedded)


# Control table
#topics_str = [str(elem) for elem in x_test[0]]
#reviews_str = [str(elem) for elem in x_test[1]]
#data = {'topics': topics_str, 'reviews': reviews_str, 'sentiments': y_test}
#table = pd.DataFrame(data=data)
#print("Control table created")

########################################################################################################################
# NN model #
########################################################################################################################
classes = ['positive', 'negative']

print("Building NN Model...")
nn_model = model(None,
                 tweet_max_length=text_max_length,
                 aspect_max_length=target_max_length,
                 noise=0.2,
                 activity_l2=0.001,
                 drop_text_rnn_U=0.2,
                 drop_text_input=0.3,
                 drop_text_rnn=0.3,
                 drop_target_rnn=0.2,
                 use_final=True,
                 bi=True,
                 final_size=64,
                 drop_final=0.5,
                 lr=0.001,
                 rnn_cells=64,
                 attention="context",
                 clipnorm=.1,
                 classes=len(classes))

nn_model.load_weights(best_model)

results = nn_model.predict(test_embedded)
print(results)

# I predict all reviews w.r.t. each topic
if not os.path.isfile("data/alberto/from_topic_to_embed.pickle"):
    print("Creating topics embedded")
    model = TFAutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
    topics_embedded = {}
    for elem in ['servizi', 'valore', 'personale', 'posizione', 'comodo', 'pulizia', 'wifi']:
        indexed = tok.vocab[elem]
        indexed = [indexed]
        model_input = numpy.array(indexed)
        aux = model.predict(model_input)[1]
        topics_embedded[elem] = aux
    pickle.dump(topics_embedded, open("data/alberto/from_topic_to_embed.pickle", "wb"))
else:
    print("Loading topics embedded")
    topics_embedded = pickle.load(open("data/alberto/from_topic_to_embed.pickle", "rb"))

results = {}
for topic in tqdm.tqdm(['servizi', 'valore', 'personale', 'posizione', 'comodo', 'pulizia', 'wifi']):
    topic_repeated = numpy.array([topics_embedded[topic] for elem in test_embedded[1]])
    results[topic] = nn_model.predict([topic_repeated, test_embedded[1]])


########################################################################################################################
# Writing results for Absita Evaluation #
########################################################################################################################

if not ABSITA:
    exit()

columns = ['sentence_id', 'cleanliness_presence', 'cleanliness_positive', 'cleanliness_negative', 'comfort_presence',
           'comfort_positive', 'comfort_negative', 'amenities_presence', 'amenities_positive', 'amenities_negative',
           'staff_presence', 'staff_positive', 'staff_negative', 'value_presence', 'value_positive', 'value_negative',
           'wifi_presence', 'wifi_positive', 'wifi_negative', 'location_presence', 'location_positive',
           'location_negative', 'other_presence', 'other_positive', 'other_negative', 'sentence']

# Create dictionary for ids of dictionary
reviews = [elem[1] for elem in x_test_base]
absita = {}
for i, review in zip(ids, reviews):
    if i.isdigit():
        absita[i] = {key: 0 for key in columns[1:]}
        absita[i]['sentence'] = review

# Dictionary to get correct name
traduction_inverted = {v: k for k, v in traduction.items()}

# Fill absita structure with results
for topic in results.keys():
    for n, i in enumerate(ids):
        if not i.isdigit():
            i = i[:-1]
        if results[topic][n] > 0.5:
            absita[i][traduction_inverted[topic]+'_presence'] = 1

# Write into file
with open('data/raw/alberto_acd_test_results.csv', 'w+') as f:
    for column in columns:
        f.write(column+';')
    f.write('\n')
    for n, i in enumerate(absita.keys()):
        f.write(i+';')
        for elem in absita[i].keys():
            f.write(str(absita[i][elem])+';')
        f.write(reviews[n])
        f.write('\n')

print("Aspect category detection for alberto evaluated")