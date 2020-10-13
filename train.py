import pickle
from keras.callbacks import ModelCheckpoint
from kutilities.helpers.data_preparation import get_class_weights2
from utilities.model import model
import tensorflow as tf
from load_dataset import load_alberto_dataset
from utilities.matrix_wv_generator import matrix_wv_generator
from utilities.embeddings_loader import load_embeddings
from load_dataset import load_w2v_dataset
from load_dataset import load_dataset

# Which task
TASK = "acd"
assert TASK == "acp" or TASK == "acd"
print("Executing " + TASK + " task")

# Which embeddings
EMB = "alberto"
assert EMB == "alberto" or EMB == "w2v"
print("Using "+EMB+" embeddings")

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to save things
best_model = "experiments/"+EMB+"/"+TASK+"/checkpoint"
history_file = "experiments/"+EMB+"/"+TASK+"/model_history.pickle"

# If w2v, load embeddings
embeddings = None
word_indices = None
if EMB == "w2v":
    embeddings, word_indices = matrix_wv_generator(load_embeddings(file="embeddings", dimension=300))
    pickle.dump(word_indices, open("experiments/w2v/"+TASK+"/model_word_indices.pickle", 'wb'))
    print("Embedding matrix and word indices generated")

# Load dataset ########################################################################################################
x_train, y_train, x_val, y_val, _, _ = load_dataset(filename="data/raw/train.csv",
                                                    text_max_length=50, target_max_length=1,
                                                    task=TASK, emb=EMB, word_indices=word_indices,
                                                    split_validation=True)
# if EMB is "w2v":
#     x_train, y_train, x_val, y_val, _, _ = load_w2v_dataset(filename="data/raw/train.csv",
#                                                            text_max_length=50, target_max_length=1,
#                                                            task=TASK, word_indices=word_indices,
#                                                            split_validation=True)
# else:
#    x_train, y_train, x_val, y_val, _, _ = load_alberto_dataset(filename="data/raw/train.csv",
#                                                                text_max_length=50, target_max_length=1,
#                                                                task=TASK, word_indices=word_indices,
#                                                                split_validation=True)
# if TASK == "acd" and EMB == "w2v":
#    x_train, y_train, x_val, y_val, _, _ = load_w2v_acd_dataset(text_max_length=text_max_length, target_max_length=1,
#                                                                task="acd", word_indices=word_indices)
# elif TASK == "acp" and EMB == "w2v":
#    x_train, y_train, x_val, y_val, _, _ = load_w2v_acp_dataset(text_max_length=text_max_length, target_max_length=1,
#                                                                task="acp", word_indices=word_indices)

# if EMB == "alberto":
#    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(text_max_length=text_max_length, target_max_length=1,
#                                                                  just_detection=(TASK == "acd"),
#                                                                  embedded=(EMB == "alberto"),
#                                                                  word_indices=word_indices)
print("Dataset BERTed")

########################################################################################################################
# NN model #
########################################################################################################################
if TASK == "acp":
    classes = ['negative', 'mixed', 'positive']
else:
    # classes = ['positive', 'negative']
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
# else:
    # We already have categories
    # class_weights = get_class_weights2(y_train, smooth_factor=0.1)

# print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
# Convert into number
# class_weights = {i: class_weights[w] for i, w in enumerate(class_weights.keys())}

history = nn_model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=100,
                       batch_size=64,
                       class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history, open(history_file, "wb"))

# loss, acc = nn_model.evaluate(x_test, y_test, verbose=2)
# print("Test set: accuracy: {:5.2f}%".format(100*acc))
