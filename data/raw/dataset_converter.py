from utilities.eng_to_ita import traduction
# From csv file to tsv file


def data_converter(in_file, output_file):

    reviews = []
    topics = []
    sentiments = []
    ids = []

    # Read csv file as list
    with open(in_file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    # Save columns name
    columns = lines[0].replace('"', '').replace("\n", '').split(sep=";", maxsplit=26)

    # Jump the header
    lines = lines[1:]

    # For each dataset line
    for line in lines:
        # Get values of columns into list
        values = line.split(sep=";", maxsplit=26)
        # Remove \n
        review = values[-1].replace("\n", '')
        # Declare lists for sentiments and topics
        topic_list = []
        sentiment_list = []
        # Save id which does not change
        review_id = values[0]
        # For each column
        for n, value in enumerate(values):
            # If this column has _presence into his name, it his a valid presence attribute
            if value == '1' and '_presence' in columns[n]:
                # Get english topic name removing _presence
                just_topic = columns[n].replace('_presence', '')
                topic_list.append(traduction[just_topic])
                # Read two next columns to understand which combination it is representing
                if values[n+1] == '1' and values[n+2] == '0':
                    sentiment_list.append(traduction['positive'])
                elif values[n+1] == '0' and values[n+2] == '1':
                    sentiment_list.append(traduction['negative'])
                elif values[n+1] == '1' and values[n+2] == '1':
                    sentiment_list.append(traduction['mixed'])
                elif values[n+1] == '0' and values[n+2] == '0':
                    # Since we excluded neutral because it has no samples, neutral = mixed
                    sentiment_list.append(traduction['mixed'])
                else:
                    raise Exception

        # If there are no topics, we must invent one to preserve the id
        if len(topic_list) == 0:
            topic_list.append(traduction['other'])
        # Insert into lists repeated values
        # There are sentiments (ACP)
        if len(sentiment_list) > 0:
            for t, s in zip(topic_list, sentiment_list):
                reviews.append(review)
                topics.append(t)
                sentiments.append(s)
                ids.append(review_id)
        # There are no sentiments (ACD)
        else:
            for t in topic_list:
                reviews.append(review)
                topics.append(t)
                sentiments.append('positive')
                ids.append(review_id)

    # Write results in tsv format inside output file
    with open(output_file, "w+") as f:
        for i, review, topic, sentiment in zip(ids, reviews, topics, sentiments):
            f.write(str(i)+'\t')
            f.write(str(topic)+'\t')
            f.write(str(sentiment)+'\t')
            f.write(str(review))
            f.write('\n')

    for elem in set(sentiments):
        print('{} appears {} times'.format(elem, sentiments.count(elem)))


files = ['test', 'train', 'absita_results_acd', 'absita_results_acp']

for file in files:
    try:
        print("Converting "+file+'.csv...')
        data_converter(file+'.csv', '../'+file+'.tsv')
    except:
        print("{} does not exists!".format(file))
        continue
