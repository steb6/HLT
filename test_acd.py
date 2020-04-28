import pickle
from utilities.dataset_loader import load_dataset
from utilities.dataset_indexxer import index_y, index_x
import pandas as pd
from utilities.model import model
import numpy
import tqdm
from utilities.eng_to_ita import traduction

# If we want to save results as Absita file
ABSITA = True

# Test file
test_file = 'test.tsv'

# Embeddings dimension
embeddings_dimension = (667566, 300)

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to load the model
best_model = "experiments/acd/model.dhf5"
best_model_word_indices = "experiments/acd/model_word_indices.pickle"

# Read word indices
with open(best_model_word_indices, 'rb') as f:
    word_indices = pickle.load(f)
word_indices_inverted = {v: k for k, v in word_indices.items()}

# Load data
testing = load_dataset(test_file)

# Set all positive
for key in testing.keys():
    value = list(testing[key])
    value[0] = 'positive'
    testing[key] = tuple(value)

# Get x, y from testing set
ids = testing.keys()
reviews = [elem[1][1] for elem in testing.values()]
testing = list(testing.values())
x_test = [elem[1] for elem in testing]
y_test = [elem[0] for elem in testing]

# Index dataset
x_test = index_x(data=x_test, target_max_length=target_max_length, text_max_length=text_max_length,
                 word_indices=word_indices)
print("x indexed")

y_test = index_y(y_test, {'positive': 1, 'negative': 0})
print("y indexed")

# Control table
topics_str = [str(elem) for elem in x_test[0]]
reviews_str = [str(elem) for elem in x_test[1]]
data = {'topics': topics_str, 'reviews': reviews_str, 'sentiments': y_test}
table = pd.DataFrame(data=data)
print("Control table created")

########################################################################################################################
# NN model #
########################################################################################################################
classes = ['negative', 'positive']

print("Building NN Model...")
nn_model = model(numpy.ndarray(embeddings_dimension),
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

# I predict all reviews w.r.t. each topic
results = {}
for topic in tqdm.tqdm(['servizi', 'valore', 'personale', 'posizione', 'comodo', 'pulizia', 'wifi']):
    topic_repeated = numpy.array([word_indices[topic] for i in range(len(x_test[1]))])
    results[topic] = nn_model.predict([topic_repeated, x_test[1]])


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
with open('data/raw/absita_results_acd.csv', 'w+') as f:
    for column in columns:
        f.write(column+';')
    f.write('\n')
    for n, i in enumerate(absita.keys()):
        f.write(i+';')
        for elem in absita[i].keys():
            f.write(str(absita[i][elem])+';')
        f.write(reviews[n])
        f.write('\n')
