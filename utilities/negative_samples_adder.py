import random


def add_negative_samples(data, categories, probability):
    augmented_data = data.copy()
    for key in data.keys():
        possible_topic = categories.copy()
        review = data[key][1][1]
        possible_indices = ['', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for idx in possible_indices:
            # find topic positive already present in data
            if key+idx in data:
                topic = data[key+idx][1][0]
                if topic in possible_topic:
                    possible_topic.remove(topic)
                possible_indices.remove(idx)
        # use topic_present and possible_indices to add negative examples with probability
        for i, topic in enumerate(possible_topic):
            if random.random() < probability:
                augmented_data[key + possible_indices[i]] = tuple(['negative', tuple([topic, review])])

    return augmented_data
