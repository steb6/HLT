from keras.callbacks import ModelCheckpoint, EarlyStopping
from kutilities.helpers.data_preparation import get_class_weights2
from utilities.model import model
import tensorflow as tf
from utilities.matrix_wv_generator import matrix_wv_generator
from utilities.embeddings_loader import load_embeddings
from utilities.load_dataset import load_dataset
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
import gc

rnn_cells = 256
final_cells = 64

# Validation too
validation = False

# Which task
TASK = "acp"
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
logs_folder = "logs/"+EMB+"/"+TASK
folder_best_model = "checkpoints/"+EMB+"/"+TASK
file_best_model = folder_best_model+"/checkpoint"
# history_file = "checkpoints/"+EMB+"/"+TASK+"/model_history.pickle"

# If w2v, load embeddings
embeddings = None
word_indices = None
if EMB == "w2v":
    embeddings, word_indices = matrix_wv_generator(load_embeddings(file="embeddings", dimension=300))
    # pickle word indices
    print("Embedding matrix and word indices generated")

# Load dataset ########################################################################################################
x_train, y_train, x_val, y_val, = load_dataset(which="train", text_max_length=50, target_max_length=1,
                                               task=TASK, emb=EMB, word_indices=word_indices)
if TASK is "acd":
    classes = ['cleanliness', 'comfort', "amenities", "staff", "value", "wifi", "location", "other"]
else:
    classes = ['negative', 'mixed', 'positive']

visualize = pd.DataFrame(columns=classes, data=y_train)
visualize.sum(axis=0).plot.bar()
plt.subplots_adjust(bottom=0.2)
plt.savefig("report/imgs/"+TASK+"_y_train_historgram")
print("Dataset Loaded")

########################################################################################################################
# Hyperparameters settings #
########################################################################################################################
HP_RNN_CELLS = hp.HParam('num_units_rnn', hp.Discrete([64, 128, 256]))
HP_FINAL_CELLS = hp.HParam('final_num_units', hp.Discrete([64, 128, 256]))
# HP_DROP_REP = hp.HParam('dropout_in', hp.RealInterval(0.1, 0.2))
# HP_DROP_OUT = hp.HParam('dropout_out', hp.RealInterval(0.2, 0.4))

METRIC_RECALL = 'recall'
METRIC_ACCURACY = 'accuracy'
EPOCHS = 'epochs'

with tf.summary.create_file_writer('logs/'+EMB+'/'+TASK+'/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_RNN_CELLS, HP_FINAL_CELLS],#  , HP_DROP_REP, HP_DROP_OUT],
        metrics=[hp.Metric(METRIC_RECALL, display_name='Recall'),
                 hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                 hp.Metric(EPOCHS, display_name='Epochs')]
    )

########################################################################################################################
# Validation settings #
########################################################################################################################
class_weights = None
if TASK == "acp":
    # Need to get index from one hot vector representation
    lab_to_cat = {'negative': 0, 'mixed': 1, 'positive': 2}
    cat_to_class_mapping = {v: k for k, v in lab_to_cat.items()}
    class_weights = get_class_weights2([list(elem).index(1) for elem in y_train], smooth_factor=0.1)
    print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
    class_weights = {i: class_weights[w] for i, w in enumerate(class_weights.keys())}


def validation_score_model(hparams, n_run):
    nn_model = model(embeddings,
                     text_max_length,
                     target_max_length,
                     len(classes),
                     TASK,
                     rnn_cells=hparams[HP_RNN_CELLS],
                     final_cells=hparams[HP_FINAL_CELLS],
                     # drop_rep=hparams[HP_DROP_REP],
                     # drop_out=hparams[HP_DROP_OUT]
                     )
    patience = 10
    epochs = 50
    stopper = EarlyStopping(monitor="val_recall",# if n_run is 0 else "val_recall_"+str(n_run),
                            min_delta=0.01, patience=patience, verbose=2, mode="max",
                            restore_best_weights=True)
    history = nn_model.fit(x_train, y_train, epochs=epochs, class_weight=class_weights, batch_size=64,
                           validation_data=(x_val, y_val), callbacks=[stopper])
    # callbacks=[
    #    tf.keras.callbacks.TensorBoard(log_dir),  # log metrics
    #    hp.KerasCallback(log_dir, hparams),  # log hparams
    # ])
    _, accuracy, recall = nn_model.evaluate(x_val, y_val)
    del nn_model

    return accuracy, recall, epochs if len(history.epoch) is epochs else len(history.epoch) - patience


def run(run_dir, hparams, n_run):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy, recall, epochs = validation_score_model(hparams, n_run)
        gc.collect()
        # To free memory for w2v model
        tf.keras.backend.clear_session()
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        tf.summary.scalar(METRIC_RECALL, recall, step=1)
        tf.summary.scalar(EPOCHS, epochs, step=1)


########################################################################################################################
# Start validation #
########################################################################################################################
session_num = 0

if validation:
    # Erase old logs
    if os.path.isdir(logs_folder):
        shutil.rmtree(logs_folder)
    os.makedirs(logs_folder)
    # Grid search
    for rnn_cells in HP_RNN_CELLS.domain.values:
        for final_cells in HP_FINAL_CELLS.domain.values:
            # for drop_rep in (HP_DROP_REP.domain.min_value, HP_DROP_REP.domain.max_value):
            # for drop_out in (HP_DROP_OUT.domain.min_value, HP_DROP_OUT.domain.max_value):
            hparams = {
                HP_RNN_CELLS: rnn_cells,
                HP_FINAL_CELLS: final_cells,
                # HP_DROP_REP: drop_rep,
                # HP_DROP_OUT: drop_out
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run(logs_folder+'/hparam_tuning/' + run_name, hparams, session_num)
            session_num += 1
    exit()


########################################################################################################################
# TRAINING BEST MODEL #
########################################################################################################################

# Erase files from older training
if os.path.isdir(folder_best_model):
    shutil.rmtree(folder_best_model)
os.makedirs(folder_best_model)

if os.path.isdir(logs_folder+'/final_training/'):
    shutil.rmtree(logs_folder+'/final_training/')
os.makedirs(logs_folder+'/final_training/')

augmented_filename = file_best_model+'_'+str(rnn_cells)+'_'+str(final_cells)
checkpointer = ModelCheckpoint(filepath=augmented_filename, monitor='val_recall',
                               mode="max", verbose=1, save_best_only=True, save_weights_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/"+EMB+"/"+TASK+"/final_training/", histogram_freq=1)

_callbacks = [tensorboard_callback, checkpointer]

nn_model = model(embeddings,
                 text_max_length,
                 target_max_length,
                 len(classes),
                 TASK,
                 rnn_cells=rnn_cells,
                 final_cells=final_cells,
                 # drop_rep=hparams[HP_DROP_REP],
                 # drop_out=hparams[HP_DROP_OUT]
                 )


history = nn_model.fit(x_train, y_train,
                       validation_data=(x_val, y_val),
                       epochs=50,
                       batch_size=64,
                       class_weight=class_weights,
                       callbacks=_callbacks)

# pickle.dump(history.history, open(history_file, "wb"))

# loss, acc = nn_model.evaluate(x_test, y_test, verbose=2)
# print("Test set: accuracy: {:5.2f}%".format(100*acc))
