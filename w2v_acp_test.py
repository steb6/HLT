import pickle
from utilities.dataset_loader import load_dataset
from utilities.dataset_indexxer import index_y, index_x
import pandas as pd
from utilities.model import model
import numpy
from utilities.eng_to_ita import traduction
from utilities.y_converter import y_label_to_category_map, y_category_to_onehot, onehot_to_category

# If we want to save results as Absita file
ABSITA = True

# Test file
test_file = 'data/test.tsv'

# Embeddings dimension
embeddings_dimension = (667566, 300)

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to load the model
best_model = "experiments/w2v/acp/checkpoint"
best_model_word_indices = "experiments/w2v/acp/model_word_indices.pickle"

# Read word indices
with open(best_model_word_indices, 'rb') as f:
    word_indices = pickle.load(f)
word_indices_inverted = {v: k for k, v in word_indices.items()}

# Load data
testing = load_dataset(test_file)

# Converting data
ids = testing.keys()
testing = list(testing.values())
x_test = [elem[1] for elem in testing]
y_test = [elem[0] for elem in testing]
print("Data rode")

# Index dataset
x_test = index_x(data=x_test, target_max_length=target_max_length, text_max_length=text_max_length,
                 word_indices=word_indices)
print("x indexed")

# Index y: one hot encoding
y_test_categories = [y_label_to_category_map[elem] for elem in y_test]
y_test_one_hot = [y_category_to_onehot[elem] for elem in y_test_categories]
y_category_to_label = {v: k for k, v in y_label_to_category_map.items()}
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
classes = ['negative', 'positive', 'mixed']

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

results = nn_model.predict(x_test)
print(results)

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

# Create dictionary for ids with dictionary for columns
absita = {}
for i in ids:
    if i.isdigit():
        absita[i] = {key: 0 for key in columns[1:]}

# Dictionary to get correct name
traduction_inverted = {v: k for k, v in traduction.items()}

# Transform x_test[0] in a list of possible topics
# from number to italian topic
topics = []
for elem in x_test[0]:
    topics.append(word_indices_inverted[elem[0]])
# from italian topic to english topic
aux = topics
topics = []
for a in aux:
    topics.append(traduction_inverted[a])

# Change results from one-hot to categories:
results_categories = onehot_to_category(results)

# Change results_categories from categories to label
results_label = []
for result in results_categories:
    results_label.append(y_category_to_label[result])

# Fill the absita structure with results
for i, result, topic in zip(ids, results_label, topics):
    if not i.isdigit():
        i = i[:-1]
    absita[i][topic+'_presence'] = 1
    if result == 'positive':
        absita[i][topic+'_positive'] = 1
    elif result == 'negative':
        absita[i][topic+'_negative'] = 1
    elif result == 'mixed':
        absita[i][topic + '_negative'] = 1
        absita[i][topic + '_positive'] = 1
    else:
        raise Exception()

# Write into file
with open('data/raw/w2v_acp_test_results.csv', 'w+') as f:
    for column in columns:
        f.write(column+';')
    f.write('\n')
    for i in absita.keys():
        # Write id
        f.write(i+';')
        # Write all other columns' value
        for elem in absita[i].keys():
            f.write(str(absita[i][elem])+';')
        f.write('\n')

print("Created result in data/raw/w2v_acp_test_results.csv")