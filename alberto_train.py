import pickle
from keras.callbacks import ModelCheckpoint
from kutilities.helpers.data_preparation import get_class_weights2
from utilities.model import model
import tensorflow as tf
from load_dataset import load_dataset

# Which task
TASK = "acd"
assert TASK == "acp" or TASK == "acd"
print("Executing " + TASK + " task")

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to save things
best_model = "experiments/alberto/"+TASK+"/checkpoint"
history_file = "experiments/alberto/"+TASK+"/model_history.pickle"

# Load dataset
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(embedded=True, text_max_length=text_max_length,
                                                              just_detection=(TASK == "acd"))
print("Dataset BERTed")

########################################################################################################################
# NN model #
########################################################################################################################
if TASK == "acp":
    classes = ['negative', 'mixed', 'positive']
else:
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
if TASK == "acp":
    lab_to_cat = {'mixed': 0, 'negative': 1, 'positive': 2}
else:
    lab_to_cat = {'positive': 1, 'negative': 0}

cat_to_class_mapping = {v: k for k, v in lab_to_cat.items()}


checkpointer = ModelCheckpoint(filepath=best_model, monitor='val_recall',
                               mode="max", verbose=1, save_best_only=True, save_weights_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/alberto/'+TASK, histogram_freq=1)

_callbacks = [tensorboard_callback, checkpointer]

########################################################################################################################
# Class weights and fitting #
########################################################################################################################
if TASK == "acp":
    # Need to get index from one hot vector representation
    class_weights = get_class_weights2([list(elem).index(1) for elem in y_train], smooth_factor=0.1)
else:
    # We already have categories
    class_weights = get_class_weights2(y_train, smooth_factor=0.1)

print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
# Convert into number
class_weights = {i: class_weights[w] for i, w in enumerate(class_weights.keys())}

history = nn_model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=100, batch_size=64, class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history, open(history_file, "wb"))

loss, acc = nn_model.evaluate(x_test, y_test, verbose=2)
print("Test set: accuracy: {:5.2f}%".format(100*acc))
