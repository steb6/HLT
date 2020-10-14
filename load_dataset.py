import os
from transformers import AutoTokenizer, TFAutoModel
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pickle
from kutilities.helpers.data_preparation import categories_to_onehot
import numpy


traduction = {'cleanliness': 'pulizia',
              'comfort': 'comodo',
              'amenities': 'servizi',
              'staff': 'personale',
              'value': 'valore',
              'wifi': 'wifi',
              'location': 'posizione',
              'other': 'altro',
              'positive': 'positive',
              'negative': 'negative',
              'neutral': 'neutral',
              'mixed': 'mixed'}


def load_dataset(which="train", text_max_length=50, target_max_length=1, task="acd", emb="alberto",
                 word_indices=None):
    # Test or data
    if which is "train":
        filename = "data/raw/train.csv"
    else:
        filename = "data/raw/test.csv"

    # First 2 change only if we use alberto, the last one changes only if we do acd
    tok = None
    model = None
    polarities = None
    if emb is "alberto":
        # Load model
        model = TFAutoModel.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
        tok = AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")
        # print(cosine_similarity(model.predict(np.array([tok.vocab["cane"]]))[1],
        #                        model.predict(np.array([tok.vocab["gatto"]]))[1]))
        # print(cosine_similarity(model.predict(np.array([tok.vocab["cane"]]))[1],
        #                         model.predict(np.array([tok.vocab["ruota"]]))[1]))

########################################################################################################################
# Read CSV #
########################################################################################################################

    with open(filename, "r", encoding='utf-8') as f:
        lines = f.readlines()
    # Save columns name
    columns = lines[0].replace('"', '').replace("\n", '').split(sep=";", maxsplit=26)
    # Jump the header
    lines = lines[1:]
    # Split line values
    lines = [elem.split(";") for elem in lines]
    # Sanitize reviews
    for line in lines:
        line[-1] = line[-1].replace('"', '').replace('\n', '').replace('.', '').replace(',', '').replace('\'', '')
    topics = []
    reviews = []
    sentiments = []
    # get reviews and topic from lines
    for line in lines:
        topic = []
        for i in range(1, 24, 3):
            topic.append(int(line[i]))
            # if this topic is present, one of the two following is the sentiment
            if int(line[i]) == 1:
                if int(line[i+1]) == 1 and int(line[i+2]) == 1:
                    sentiments.append(columns[i+1].split('_')[0] + '_mixed')
                elif int(line[i+1]) == 1:
                    sentiments.append(columns[i+1])
                else:
                    sentiments.append(columns[i+2])
        # Add to reviews the review the correct number of times for the task
        if task is "acd":
            reviews.append(line[-1])
        if task is "acp":
            for elem in topic:
                if elem == 1:
                    reviews.append(line[-1])
        topics.append(topic)

########################################################################################################################
# Index or embed #
########################################################################################################################

    if emb is "w2v":
        x_train = numpy.zeros((len(reviews), text_max_length), dtype=int)
        known = 0
        unknown = 0
        for i, review in enumerate(reviews):
            for j, word in enumerate(review.split(' ')):
                if j == text_max_length:
                    break
                if word in word_indices:
                    x_train[i][j] = word_indices[word]
                    known += 1
                else:
                    x_train[i][j] = word_indices["<unk>"]
                    unknown += 1
        print("Percentage of known words: ", (known/(known+unknown))*100)
        y_train = np.array(topics)

        # If polarity detection, we need to change the y from topics to sentiments and append topics to x
        if task == "acp":
            topics = []
            polarities = []
            for sentiment in sentiments:
                topic, polarity = sentiment.split('_')
                polarities.append(0 if polarity == 'negative' else 1 if polarity == 'mixed' else 2)
                topics.append([word_indices[traduction[topic]]])
    else:  # alberto
        if os.path.isfile("data/alberto/" + task + "/" + which + "_reviews_embedded.pickle"):
            x_train = pickle.load(open("data/alberto/" + task + "/" + which + "_reviews_embedded.pickle", "rb"))
        else:
            x_train = numpy.zeros((len(reviews), text_max_length, 768), dtype=float)
            print("Embedding reviews...")
            for i, review in tqdm(enumerate(reviews), total=len(reviews)):
                ind_review = [tok.vocab[token] for token in tok.tokenize(review)]
                while len(ind_review) < text_max_length:
                    ind_review.append(0)
                if len(ind_review) > text_max_length:
                    ind_review = ind_review[:text_max_length]
                emb_review = model.predict(ind_review)[1]
                x_train[i] = emb_review
            pickle.dump(x_train, open("data/alberto/" + task + "/" + which + "_reviews_embedded.pickle", "wb"))
        y_train = np.array(topics)

        # If polarity detection, we need to change the y from topics to sentiments and append topics to x
        if task == "acp":
            if os.path.isfile("data/alberto/" + task + "/" + which + "_topics_embedded.pickle") \
                    and os.path.isfile("data/alberto/" + task + "/" + which + "_polarities_embedded.pickle"):
                topics = pickle.load(open("data/alberto/" + task + "/" + which + "_topics_embedded.pickle", "rb"))
                polarities = pickle.load(open("data/alberto/" + task + "/" + which + "_polarities_embedded.pickle", "rb"))
            else:
                topics = numpy.zeros((len(reviews), target_max_length, 768), dtype=float)
                polarities = []
                print("Embedding topics...")
                for i, sentiment in tqdm(enumerate(sentiments), total=len(sentiments)):
                    topic, polarity = sentiment.split('_')
                    polarities.append(0 if polarity == 'negative' else 1 if polarity == 'mixed' else 2)
                    topics[i] = model.predict([tok.vocab[traduction[topic]]])[1]
                pickle.dump(topics, open("data/alberto/" + task + "/" + which + "_topics_embedded.pickle", "wb"))
                pickle.dump(polarities, open("data/alberto/" + task + "/" + which + "_polarities_embedded.pickle", "wb"))

########################################################################################################################
# Format data in the right way #
########################################################################################################################

    if task is "acp":
        final_x = []
        for i, j in zip(np.array(topics), x_train):
            final_x.append((i, j))
        x_train = final_x
        y_train = polarities

    # We can stratify over acp, but not over acd because the combination of classes are few
    if which is "train":
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3,
                                                          stratify=None if task is "acd" else y_train,
                                                          random_state=42)
    else:
        x_val = y_val = None

    # If we have to do detection, we finished
    if task is "acd":
        return x_train, y_train, x_val, y_val

    # If we have to do polarities, we need to change the input format
    final_topics = []
    final_reviews = []
    for elem in x_train:
        final_topics.append(elem[0])
        final_reviews.append(elem[1])
    x_train = [np.array(final_topics), np.array(final_reviews)]
    y_train = np.array(categories_to_onehot(y_train))

    if x_val is not None and y_val is not None:
        final_topics = []
        final_reviews = []
        for elem in x_val:
            final_topics.append(elem[0])
            final_reviews.append(elem[1])
        x_val = [np.array(final_topics), np.array(final_reviews)]
        y_val = np.array(categories_to_onehot(y_val))

    return x_train, y_train, x_val, y_val
