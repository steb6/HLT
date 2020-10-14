import pickle
from utilities.model import model
from utilities.matrix_wv_generator import matrix_wv_generator
from utilities.embeddings_loader import load_embeddings
from load_dataset import load_dataset
import numpy as np

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
x_test, y_test, _, _ = load_dataset(which="test", text_max_length=50, target_max_length=1,
                                    task=TASK, emb=EMB, word_indices=word_indices)

print("Dataset BERTed")

########################################################################################################################
# NN model #
########################################################################################################################
if TASK == "acp":
    classes = ['negative', 'mixed', 'positive']
else:
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

# print(nn_model.summary())
nn_model.load_weights(best_model)
print("Model loaded!")
results = nn_model.predict(x_test)

# Load reviews id
with open("data/raw/test.csv", "r", encoding='utf-8') as f:
    lines = f.readlines()

if TASK is "acd":
    # Create results structure
    results_rounded = np.zeros((len(results), 24), dtype=int)
    for i, line in enumerate(results):
        for j, elem in enumerate(line):
            if elem > 0.5:
                results_rounded[i][j*3] = 1
    columns = lines[0]
    ids = [line.split(";")[0] for line in lines[1:]]  # skip the header
else:
    # Transform results in polarities
    polarities = []
    for result in results:
        result = list(result)
        polarity = result.index(max(result))
        polarities.append(polarity)

    ids = []
    results_rounded = np.zeros((len(lines), 24), dtype=int)
    for i, line in enumerate(lines[1:]):
        values = line.split(";")
        ids.append(values[0])
        columns = values[1:]
        topic = []
        for elem in range(0, 24, 3):
            if int(columns[elem]) == 1:
                results_rounded[i][elem] = 1
                sentiment = polarities.pop(0)
                if sentiment is 2:
                    results_rounded[i][elem + 1] = 1
                elif sentiment is 0:
                    results_rounded[i][elem + 2] = 1
                else:
                    results_rounded[i][elem + 1] = 1
                    results_rounded[i][elem + 2] = 1

with open("data/" + TASK + "_" + EMB + "_results.csv", "w") as f:
    f.write(lines[0])
    for i, line in zip(ids, results_rounded):
        f.write(i)
        f.write(";")
        for elem in line:
            f.write(str(elem))
            f.write(";")
        f.write("\n")

print("Results are available in: " + "data/" + TASK + "_" + EMB + "_results.csv")