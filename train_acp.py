from utilities.embeddings_loader import load_embeddings
from utilities.matrix_wv_generator import matrix_wv_generator
from utilities.dataset_loader import load_dataset
from utilities.dataset_indexxer import index_x
from utilities.model import model
from kutilities.helpers.data_preparation import get_class_weights2
from sklearn.metrics import precision_score, recall_score, f1_score
from kutilities.callbacks import MetricsCallback, WeightsCallback, PlottingCallback
from keras.callbacks import ModelCheckpoint
import pickle
from sklearn.model_selection import train_test_split
from kutilities.helpers.data_preparation import labels_to_categories, categories_to_onehot, get_labels_to_categories_map
from utilities.y_converter import y_label_to_category_map

########################################################################################################################
# Setting variables #
########################################################################################################################

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to save the model
best_model = "experiments/acp/model.dhf5"
best_model_word_indices = "experiments/acp/model_word_indices.pickle"
history_file = "experiments/acp/model_history.pickle"
plot_file = "acp/plots"

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

# Remove ids converting dict into list
training = list(training.values())

# Get x and y from training list
x = [elem[1] for elem in training]
y = [elem[0] for elem in training]

# Divide training set into training and validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

# Remove id converting dict into list
testing = list(testing.values())

# Get x and y from testing list
x_test = [elem[1] for elem in testing]
y_test = [elem[0] for elem in testing]

# Print statistics
print("Rode {} training reviews, {} validation reviews and {} testing reviews".format(
    len(x_train), len(x_val), len(x_test)))

# Index dataset: word_indices
# Senza lower e remove l', il 15% era sconosciuto
# ora il 5%
x_train = index_x(data=x_train, target_max_length=target_max_length, text_max_length=text_max_length,
                  word_indices=word_indices)
x_val = index_x(data=x_val, target_max_length=target_max_length, text_max_length=text_max_length,
                word_indices=word_indices)
x_test = index_x(data=x_test, target_max_length=target_max_length, text_max_length=text_max_length,
                 word_indices=word_indices)
print("x indexed")

# Index y: one hot encoding
y_train_categories = labels_to_categories(y_train)
y_train_one_hot = categories_to_onehot(y_train_categories)
y_train_map = get_labels_to_categories_map(y_train)

y_val_categories = labels_to_categories(y_val)
y_val_one_hot = categories_to_onehot(y_val_categories)
y_val_map = get_labels_to_categories_map(y_val)

y_test_categories = labels_to_categories(y_test)
y_test_one_hot = categories_to_onehot(y_test_categories)
y_test_map = get_labels_to_categories_map(y_test)

# Assert values are categorized in the same way
assert y_train_map == y_val_map == y_test_map == y_label_to_category_map
lab_to_cat = y_train_map
print("y indexed")


########################################################################################################################
# NN model #
########################################################################################################################
classes = ['negative', 'mixed', 'positive']

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

# Retrieve class name
cat_to_class_mapping = {v: k for k, v in lab_to_cat.items()}

# Check that class weights correspond to right categorical value
assert y_train_map == y_val_map == y_test_map == lab_to_cat

# Define metrics
metrics = {
    "recall": (lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')),
    "precision": (lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro')),
    "f1-score": (lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'))
}

# Define datasets for metrics evaluation
_datasets = {"1-train": ((x_train, y_train_one_hot),),
             "2-val": (x_val, y_val_one_hot),
             "3-test": (x_test, y_test_one_hot)}

# Define callbacks
metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
weights = WeightsCallback(parameters=["W"], stats=["raster", "mean", "std"])
# f1-score: 0.7673
plotting = PlottingCallback(grid_ranges=(0.5, 1), height=4, benchmarks={"ρ": 0.8264, "r": 0.716}, plot_name=plot_file)
checkpointer = ModelCheckpoint(filepath=best_model, monitor='2-val.recall',
                               mode="max", verbose=1, save_best_only=True)

_callbacks = [metrics_callback, plotting, weights, checkpointer]

########################################################################################################################
# Class weights and fitting #
########################################################################################################################
class_weights = get_class_weights2([elem for elem in y_train_categories], smooth_factor=0.1)

print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
# Convert into number
class_weights = {i: class_weights[w] for i, w in enumerate(class_weights.keys())}

history = nn_model.fit(x_train, y_train_one_hot,
                       validation_data=(x_val, y_val_one_hot),
                       epochs=100, batch_size=64, class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history, open(history_file, "wb"))
