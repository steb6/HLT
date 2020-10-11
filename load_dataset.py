import os
from transformers import AutoConfig
from utilities.dataset_loader import load_train_test_files
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import pickle
from kutilities.helpers.data_preparation import labels_to_categories, categories_to_onehot, get_labels_to_categories_map
from utilities.negative_samples_adder import add_negative_samples


def load_dataset(embedded=False, text_max_length=50, just_detection=False):

    TASK = 'acd' if just_detection else 'acp'

    if not os.path.isfile("data/alberto/"+TASK+"/test_embedded.pickle"):
        print("alBERTed dataset not found, loading dataset and alBERTo model...")

        # Load model
        model = AutoConfig.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
        tok = AutoConfig.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

        # Load dataset
        training = load_train_test_files('data/train.tsv')
        testing = load_train_test_files('data/test.tsv')

        # Add negative samples if we are loading ACD
        if just_detection:
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

        pickle.dump(train_embedded, open("data/alberto/"+TASK+"/train_embedded.pickle", "wb"))
        pickle.dump(val_embedded, open("data/alberto/"+TASK+"/val_embedded.pickle", "wb"))
        pickle.dump(test_embedded, open("data/alberto/"+TASK+"/test_embedded.pickle", "wb"))
        pickle.dump(lab_to_cat, open("data/alberto/"+TASK+"/lab_to_cat.pickle", "wb"))
        pickle.dump(y_train_categories, open("data/alberto/"+TASK+"/y_train_categories.pickle", "wb")) # for class weights

    else: # we have the embeddings ready,w I just need to load them ###################################################

        print("Pickle files for embeddings found!")
        train_embedded = pickle.load(open("data/alberto/"+TASK+"/train_embedded.pickle", "rb"))
        val_embedded = pickle.load(open("data/alberto/"+TASK+"/val_embedded.pickle", "rb"))
        test_embedded = pickle.load(open("data/alberto/"+TASK+"/test_embedded.pickle", "rb"))
        lab_to_cat = pickle.load(open("data/alberto/"+TASK+"/lab_to_cat.pickle", "rb"))
        y_train_categories = pickle.load(open("data/alberto/"+TASK+"/y_train_categories.pickle", "rb")) # for class weights

    # Transform data to be model input
    if just_detection:
        train_embedded = train_embedded[:int(len(train_embedded)/2)]
        val_embedded = val_embedded[:int(len(val_embedded)/2)]
        test_embedded = test_embedded[:int(len(test_embedded)/2)]
        # Transform data to be model input
        x_train = [elem[1] for elem in train_embedded]
        y_train = [elem[0] for elem in train_embedded]
        del train_embedded
        x_train = [np.array([elem[0] for elem in x_train]), np.array([elem[1] for elem in x_train])]

        x_val = [elem[1] for elem in val_embedded]
        y_val = [elem[0] for elem in val_embedded]
        del val_embedded
        x_val = [np.array([elem[0] for elem in x_val]), np.array([elem[1] for elem in x_val])]
        x_test = []
        y_test = []
        y_train = np.array([elem[0] for elem in y_train])
        y_val = np.array([elem[0] for elem in y_val])
    else:
        x_train = [elem[1] for elem in train_embedded]
        x_train = [np.array([elem[0] for elem in x_train]), np.array([elem[1] for elem in x_train])]
        y_train = np.array([elem[0] for elem in train_embedded])

        x_val = [elem[1] for elem in val_embedded]
        x_val = [np.array([elem[0] for elem in x_val]), np.array([elem[1] for elem in x_val])]
        y_val = np.array([elem[0] for elem in val_embedded])

        x_test = [elem[1] for elem in test_embedded]
        x_test = [np.array([elem[0] for elem in x_test]), np.array([elem[1] for elem in x_test])]
        y_test = np.array([elem[0] for elem in test_embedded])

    return x_train, y_train, x_val, y_val, x_test, y_test
