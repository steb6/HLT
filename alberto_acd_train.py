from transformers import AutoTokenizer, TFAutoModel
import numpy as np
from kutilities.helpers.data_preparation import labels_to_categories, categories_to_onehot, get_labels_to_categories_map
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from keras.callbacks import ModelCheckpoint
from kutilities.helpers.data_preparation import get_class_weights2
import os
from utilities.model import model
from utilities.negative_samples_adder import add_negative_samples
import tensorflow as tf
from utilities.dataset_indexxer import index_y
from load_dataset import load_dataset
# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to save things
best_model = "experiments/alberto/acd/checkpoint"
history_file = "experiments/alberto/acd/model_history.pickle"

x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(embedded=True, text_max_length=text_max_length, just_detection=True)
print("Dataset BERTed")

########################################################################################################################
# NN model #
########################################################################################################################
y_train = np.array([elem[0] for elem in y_train])
y_val = np.array([elem[0] for elem in y_val])

# y_train = index_y(y_train, {'positive': 1, 'negative': 0})
# y_val = index_y(y_val, {'positive': 1, 'negative': 0})
# y_test = index_y(y_test, {'positive': 1, 'negative': 0})

classes = ['positive', 'negative']

print("Building NN Model...")
nn_model = model(None,
                 text_max_length,
                 target_max_length,
                 len(classes),
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
                 clipnorm=.1)

print(nn_model.summary())

########################################################################################################################
# Callbacks #
########################################################################################################################

# Retrieve class name
lab_to_cat = {'positive': 1, 'negative': 0}
cat_to_class_mapping = {v: k for k, v in lab_to_cat.items()}


checkpointer = ModelCheckpoint(filepath=best_model, monitor='val_recall',
                               mode="max", verbose=1, save_best_only=True, save_weights_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/alberto/acd', histogram_freq=1)

_callbacks = [tensorboard_callback, checkpointer]

########################################################################################################################
# Class weights and fitting #
########################################################################################################################
class_weights = get_class_weights2(y_train, smooth_factor=0.1)

print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
# Convert into number
class_weights = {i: class_weights[w] for i, w in enumerate(class_weights.keys())}

# Convert y from 2 to 1
# y_train = np.array([1. if elem[0]==1. else 0. for elem in y_train])

history = nn_model.fit(x_train, y_train, # si aspetta lista di due ndarray 6238x1 e 6238x50
                       validation_data=(x_val, y_val),
                       epochs=100, batch_size=64,
                       # class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history, open(history_file, "wb"))

# loss, acc = nn_model.evaluate(x_test, y_test, verbose=2)
# print("Test set: accuracy: {:5.2f}%".format(100*acc))
