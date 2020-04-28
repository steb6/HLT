import html
# From tsv to dict of form {id: sentiment, (topic, review)...} and does some preprocessing with clean_text


def clean_text(text):
    text = text.rstrip()
    text = text.replace('\"', '')
    text = text.replace(',', '')
    text = text.replace('.', '')
    text = text.lower()
    text = text.replace('l\'', ' ')
    text = html.unescape(text)
    text = ' '.join([word for word in text.split()])
    return text


def load_dataset(filename):

    data = {}
    filename = 'data/'+filename
    print("Parsing file:", filename)

    for line_id, line in enumerate(open(filename, "r", encoding="utf-8").readlines()):
        try:
            # create list of column and get id
            columns = line.rstrip().split('\t')
            review_id = columns[0]

            topic = clean_text(columns[1])
            if not isinstance(topic, str) or "None" in topic:
                print(review_id, topic)
            sentiment = columns[2]
            text = clean_text(" ".join(columns[3:]))

            # Insert review, if id is present, then add literal
            lit = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
            if text != "Not Available":
                if review_id in data:
                    review_id = review_id + (lit[0])
                    indx = 0
                    while review_id in data:
                        indx += 1
                        review_id = review_id[:-1]
                        review_id = review_id + lit[indx]
                    data[review_id] = (sentiment, (topic, text))
                else:
                    data[review_id] = (sentiment, (topic, text))

        except Exception:
            print("\nWrong format in line:{} in file:{}".format(line_id, filename))
            raise Exception

    return data
