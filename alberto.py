from transformers import AutoTokenizer, TFAutoModel
import numpy as np
from utilities.dataset_loader import load_dataset
from kutilities.helpers.data_preparation import labels_to_categories, categories_to_onehot, get_labels_to_categories_map
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score
from kutilities.callbacks import MetricsCallback, WeightsCallback, PlottingCallback
from keras.callbacks import ModelCheckpoint
from kutilities.helpers.data_preparation import get_class_weights2
import os
from utilities.model import model
import tensorflow as tf

# Input dimensions
text_max_length = 50
target_max_length = 1

# Where to save things
best_model = "experiments/alberto/model.dhf5"
history_file = "experiments/alberto/model_history.pickle"
plot_file = "experiments/alberto/plots"

# Check if pickle files exist, otherwise, construct it
if not os.path.isfile("alberto/test_embedded.pickle"):
    print("alBERTed dataset not found, loading dataset and alberto to generate it...")

    # Load model
    model = TFAutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
    tok = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

    # Load dataset
    training = load_dataset('data/train.tsv')
    testing = load_dataset('data/test.tsv')

    # Remove ids converting dict into list
    training = list(training.values())
    testing = list(testing.values())

    # Get x and y from training list
    x_train = [elem[1] for elem in training]
    y_train = [elem[0] for elem in training]
    x_test = [elem[1] for elem in testing]
    y_test = [elem[0] for elem in testing]

    # Divide training set into training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, stratify=y_train, random_state=42)

    # test
    print(cosine_similarity(model.predict(np.array([tok.vocab["cane"]]))[1],
                            model.predict(np.array([tok.vocab["gatto"]]))[1]))
    print(cosine_similarity(model.predict(np.array([tok.vocab["cane"]]))[1],
                            model.predict(np.array([tok.vocab["ruota"]]))[1]))

    # Process x
    print("Preparing dataset...")

    def index_pad_embed(x, y):
        results = []
        for couple, sent in tqdm(zip(x, y), total=len(x)):
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

    # Process y
    y_train_categories = labels_to_categories(y_train)
    y_train_one_hot = categories_to_onehot(y_train_categories)
    y_train_map = get_labels_to_categories_map(y_train)

    y_val_categories = labels_to_categories(y_val)
    y_val_one_hot = categories_to_onehot(y_val_categories)
    y_val_map = get_labels_to_categories_map(y_val)

    y_test_categories = labels_to_categories(y_test)
    y_test_one_hot = categories_to_onehot(y_test_categories)
    y_test_map = get_labels_to_categories_map(y_test)

    # train
    train_embedded = index_pad_embed(x_train, y_train_one_hot)
    # val
    val_embedded = index_pad_embed(x_val, y_val_one_hot)
    # test
    test_embedded = index_pad_embed(x_test, y_test_one_hot)

    assert y_train_map == y_val_map == y_test_map
    lab_to_cat = y_train_map

    pickle.dump(train_embedded, open("alberto/train_embedded.pickle", "wb"))
    pickle.dump(val_embedded, open("alberto/val_embedded.pickle", "wb"))
    pickle.dump(test_embedded, open("alberto/test_embedded.pickle", "wb"))
    pickle.dump(lab_to_cat, open("alberto/lab_to_cat.pickle", "wb"))
    pickle.dump(y_train_categories, open("alberto/y_train_categories.pickle", "wb")) # for class weights

else: # we have the embeddings ready,w I just need to load them

    print("Pickle files for embeddings found!")
    train_embedded = pickle.load(open("alberto/train_embedded.pickle", "rb"))
    val_embedded = pickle.load(open("alberto/val_embedded.pickle", "rb"))
    test_embedded = pickle.load(open("alberto/test_embedded.pickle", "rb"))
    lab_to_cat = pickle.load(open("alberto/lab_to_cat.pickle", "rb"))
    y_train_categories = pickle.load(open("alberto/y_train_categories.pickle", "rb"))

x_train = [elem[1] for elem in train_embedded]
y_train = [elem[0] for elem in train_embedded]
x_val = [elem[1] for elem in val_embedded]
y_val = [elem[0] for elem in val_embedded]
x_test = [elem[1] for elem in test_embedded]
y_test = [elem[0] for elem in test_embedded]

print("Dataset BERTed")

########################################################################################################################
# NN model #
########################################################################################################################
classes = ['negative', 'mixed', 'positive']

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
lab_to_cat = {'mixed': 0, 'negative': 1, 'positive': 2}
cat_to_class_mapping = {v: k for k, v in lab_to_cat.items()}

# Check that class weights correspond to right categorical value
#assert y_train_map == y_val_map == y_test_map == lab_to_cat

# Define metrics
metrics = {
    "recall": (lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')),
    "precision": (lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro')),
    "f1-score": (lambda y_true, y_pred: f1_score(y_true, y_pred, average='micro'))
}

# Define datasets for metrics evaluation
_datasets = {"1-train": ((x_train, y_train),),
             "2-val": (x_val, y_val),
             "3-test": (x_test, y_test)}

# Define callbacks
metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
weights = WeightsCallback(parameters=["W"], stats=["raster", "mean", "std"])
# f1-score: 0.7673
# plotting = PlottingCallback(grid_ranges=(0.5, 1), height=4, benchmarks={"ρ": 0.8264, "r": 0.716}, plot_name=plot_file)
checkpointer = ModelCheckpoint(filepath=best_model, monitor='2-val.recall',
                               mode="max", verbose=1, save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                                      profile_batch=5)

#_callbacks = [metrics_callback, tensorboard_callback, weights, checkpointer]
_callbacks = [tensorboard_callback]

########################################################################################################################
# Class weights and fitting #
########################################################################################################################
class_weights = get_class_weights2([elem for elem in y_train_categories], smooth_factor=0.1)

print("Class weights:", {cat_to_class_mapping[c]: w for c, w in class_weights.items()})
# Convert into number
class_weights = {i: class_weights[w] for i, w in enumerate(class_weights.keys())}

x_topic = np.array([elem[0] for elem in x_train])
x_review = np.array([elem[1] for elem in x_train])
x_train = [x_topic, x_review]

x_val_topic = np.array([elem[0] for elem in x_val])
x_val_review = np.array([elem[1] for elem in x_val])
x_val = [x_val_topic, x_val_review]

y_train = np.array(y_train)
y_val = np.array(y_val)

history = nn_model.fit(x_train, y_train, # si aspetta lista di due ndarray 6238x1 e 6238x50
                       validation_data=(x_val, y_val),
                       epochs=100, batch_size=64, class_weight=class_weights,
                       callbacks=_callbacks)

pickle.dump(history.history, open(history_file, "wb"))
