import numpy
# Index data with a word_indices dictionary. Also print percentage of unknown words


def index_x(data=None, target_max_length=None, text_max_length=None, word_indices=None):
    # Count how many unknown token
    known = 0
    unknown = 0
    unknown_words = []

    # Create final x lists
    topics_ind = numpy.zeros((len(data), target_max_length), dtype=int)
    reviews_ind = numpy.zeros((len(data), text_max_length), dtype=int)

    # Iterate on lines, I get line number
    for i, line in enumerate(data):
        topic = line[0]
        review = line[1]
        # Index topic and insert, iterate on columns, I get column number
        for j, word in enumerate(topic.split(' ')):
            if j == target_max_length:
                break
            if word in word_indices:
                topics_ind[i][j] = word_indices[word]
            else:
                topics_ind[i][j] = word_indices["<unk>"]
        # Index review and insert, iterate on columns, I get column number
        for j, word in enumerate(review.split(' ')):
            if j == text_max_length:
                break
            if word in word_indices:
                reviews_ind[i][j] = word_indices[word]
                known += 1
            else:
                reviews_ind[i][j] = word_indices["<unk>"]
                unknown += 1
                unknown_words.append(word)
    print("Known token: {}, unknown token: {}, percentage of unknown: {}".format(known, unknown,
                                                                                 (unknown/(known+unknown))))
    # with open("unknown_words.txt", "w+") as f:
    #     for word in unknown_words:
    #         f.write(word+'\n')

    return [topics_ind, reviews_ind]


def index_y(data=None, transformation=None):
    y_ind = []
    for i, elem in enumerate(data):
        y_ind.append(transformation[elem])
    return numpy.array(y_ind)
