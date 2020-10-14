import pickle
from keras.callbacks import ModelCheckpoint
from kutilities.helpers.data_preparation import get_class_weights2
from utilities.model import model
import tensorflow as tf
from utilities.matrix_wv_generator import matrix_wv_generator
from utilities.embeddings_loader import load_embeddings
from utilities.load_dataset import load_dataset
import shutil
import os

# Which task
TASK = "acd"
assert TASK == "acp" or TASK == "acd"
print("Executing " + TASK + " task")

# Which embeddings
EMB = "w2v"
assert EMB == "alberto" or EMB == "w2v"
print("Using "+EMB+" embeddings")

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to save things
best_model = "checkpoints/"+EMB+"/"+TASK+"/checkpoint"
history_file = "checkpoints/"+EMB+"/"+TASK+"/model_history.pickle"

# If w2v, load embeddings
embeddings = None
word_indices = None
if EMB == "w2v":
    embeddings, word_indices = matrix_wv_generator(load_embeddings(file="embeddings", dimension=300))
    print("Embedding matrix and word indices generated")

# Load dataset ########################################################################################################
x_train, y_train, x_val, y_val, = load_dataset(which="train", text_max_length=50, target_max_length=1,
                                               task=TASK, emb=EMB, word_indices=word_indices)

print("Dataset BERTed")

########################################################################################################################
# NN model #
########################################################################################################################
if TASK == "acp":
    classes = ['negative', 'mixed', 'positive']
else:
    classes = ['cleanliness', 'comfort', "amenities", "staff", "value", "wifi", "location", "other"]

print("Building NN Model...")
nn_model = model(embeddings,
                 text_max_length,
                 target_max_length,
                 len(classes),
                 TASK,
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
# Erase files from older training
shutil.rmtree("checkpoints/"+EMB+"/"+TASK)
os.makedirs("checkpoints/"+EMB+"/"+TASK)
shutil.rmtree("logs/"+EMB+"/"+TASK)
os.makedirs("logs/"+EMB+"/"+TASK)

checkpointer = ModelCheckpoint(filepath=best_model, monitor='val_recall',
                               mode="max", verbose=1, save_best_only=True, save_weights_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/"+EMB+"/"+TASK, histogram_freq=1)

_callbacks = [tensorboard_callback, checkpointer]

########################################################################################################################
# Class weights and fitting #
########################################################################################################################
class_weights = None
if TASK == "acp":
    # Need to get index from one hot vector representation
    lab_to_cat = {'negative': 0, 'mixed': 1, 'positive': 2}
    cat_to_class_mapping = {v: k for k, v in lab_to_cat.items()}
    class_weights = get_class_weights2([list(elem).index(1) for elem in y_train], smooth_factor=0.1)
    print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
    class_weights = {i: class_weights[w] for i, w in enumerate(class_weights.keys())}

history = nn_model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=50,
                       batch_size=64,
                       class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history, open(history_file, "wb"))

# loss, acc = nn_model.evaluate(x_test, y_test, verbose=2)
# print("Test set: accuracy: {:5.2f}%".format(100*acc))
