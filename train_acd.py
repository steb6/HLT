from utilities.embeddings_loader import load_embeddings
from utilities.matrix_wv_generator import matrix_wv_generator
from utilities.dataset_loader import load_dataset
from utilities.dataset_indexxer import index_x, index_y
from utilities.model import model
from utilities.negative_samples_adder import add_negative_samples
from kutilities.helpers.data_preparation import get_labels_to_categories_map, get_class_weights2
from sklearn.metrics import precision_score, recall_score, f1_score
from kutilities.callbacks import MetricsCallback, WeightsCallback, PlottingCallback
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf

########################################################################################################################
# Setting variables #
########################################################################################################################

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to save the model
best_model = "experiments/acd/model.dhf5"
best_model_word_indices = "experiments/acd/model_word_indices.pickle"
history_file = "experiments/acd/model_history.pickle"
plot_file = "acd/plots"

# Embeddings files
embeddings_file = "embeddings"
embeddings_dimension = 300

# Data name
data_file = 'data/train.tsv'
test_file = 'data/test.tsv'

########################################################################################################################
# Setting on embeddings and dataset #
########################################################################################################################

# Read embeddings
embeddings_dict = load_embeddings(embeddings_file, embeddings_dimension)
print("Found {} word vector".format(len(embeddings_dict)))

# Generate embeddings matrix and word indices dictionary
embeddings_matrix, word_indices = matrix_wv_generator(embeddings_dict)
pickle.dump(word_indices, open(best_model_word_indices, 'wb'))
print("Embedding matrix and word indices generated")

# Read dataset
training = load_dataset(data_file)
testing = load_dataset(test_file)

# Set all positive
for key in training.keys():
    value = list(training[key])
    value[0] = 'positive'
    training[key] = tuple(value)

for key in testing.keys():
    value = list(testing[key])
    value[0] = 'positive'
    testing[key] = tuple(value)

# Add negative samples
categories = list(set([elem[1][0] for elem in list(training.values())]))
training = add_negative_samples(training, categories, 0.2)
testing = add_negative_samples(testing, categories, 0.2)

# Get x, y
training = list(training.values())
x = [elem[1] for elem in training]
y = [elem[0] for elem in training]

# Divide into training and validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# Get x, y from testing set
testing = list(testing.values())
x_test = [elem[1] for elem in testing]
y_test = [elem[0] for elem in testing]

# Print control
print("Rode {} training reviews, {} validation reviews and {} testing reviews".format(
    len(x_train), len(x_val), len(x_test)))

# Index dataset
x_train = index_x(data=x_train, target_max_length=target_max_length, text_max_length=text_max_length,
                  word_indices=word_indices)
x_val = index_x(data=x_val, target_max_length=target_max_length, text_max_length=text_max_length,
                word_indices=word_indices)
x_test = index_x(data=x_test, target_max_length=target_max_length, text_max_length=text_max_length,
                 word_indices=word_indices)
print("x indexed")

y_train = index_y(y_train, {'positive': 1, 'negative': 0})
y_val = index_y(y_val, {'positive': 1, 'negative': 0})
y_test = index_y(y_test, {'positive': 1, 'negative': 0})
print("y indexed")


########################################################################################################################
# NN model #
########################################################################################################################
classes = ['negative', 'positive']

print("Building NN Model...")
nn_model = model(embeddings_matrix,
                 tweet_max_length=text_max_length,
                 aspect_max_length=target_max_length,
                 noise=0.2,
                 activity_l2=0.001,
                 drop_text_rnn_U=0.2,
                 drop_text_input=0.3,
                 drop_text_rnn=0.3,
                 drop_target_rnn=0.2,
                 final_size=64,
                 drop_final=0.5,
                 lr=0.001,
                 rnn_cells=64,
                 clipnorm=.1,
                 classes=len(classes))

print(nn_model.summary())

########################################################################################################################
# Callbacks #
########################################################################################################################

# define metrics and class weights
cat_to_class_mapping = {v: k for k, v in get_labels_to_categories_map(classes).items()}
metrics = {
    "recall": (lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')),
    "precision": (lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro')),
    "f1-score": (lambda y_true, y_pred: f1_score(y_true, y_pred))
}

_datasets = {"1-train": ((x_train, y_train),), "2-val": (x_val, y_val), "3-test": (x_test, y_test)}

metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
weights = WeightsCallback(parameters=["W"], stats=["raster", "mean", "std"])

# This function is made s.t. it can draw only 2 benchmarks: use "f1": 0.8108 if you want
plotting = PlottingCallback(grid_ranges=(0.5, 1), height=4, benchmarks={"ρ": 0.8397, "r": 0.7837}, plot_name=plot_file)
checkpointer = ModelCheckpoint(filepath=best_model, monitor='2-val.recall',
                               mode="max", verbose=1, save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                                      profile_batch=5)

_callbacks = [metrics_callback, tensorboard_callback, weights, checkpointer]

########################################################################################################################
# Class weights and fitting #
########################################################################################################################
class_weights = get_class_weights2(y_train, smooth_factor=0)

print("Class weights:",
      {cat_to_class_mapping[c]: w for c, w in class_weights.items()})

history = nn_model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=50, batch_size=64, class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history, open(history_file, "wb"))
