To download embeddings: http://hlt.isti.cnr.it/wordembeddings/
Instruction:
- The data folder contains the datasets, inside the raw folder you can find the csv format and a dataset converter, which transform the csv file into  a tsv file in the data folder, which is the file that we are gonna use for training and testing
- The embeddings folder contains a raw folder, which contains the compressed embedding and a script to extract them, the embeddings folder contains a txt, which is extracted with the script, and a pickle file, that will be generated at first training to speed up weights loading
- The experiments folder contains two folders, one for task, and each folder contains weights in .hdf5 format, the training history in a pickle format, the word_indices to index correctly the words, and the training plot
- The utilities folder contains scripts used in training and testing. I did not create class for simplicity, those folders just contains the definition of some useful function

I had 2 files: the training set and the official test set. I divided the training into training and validation and I use test set only for benchmarks.
Correct usage is:
- Download embeddings and extract them with the script
- Download the datasets and extract them with dataset_converter (must change file path)
- Execute train_acd and train_acp to train the models
- Execute test_acd, extract the created dataset in csv format with dataset_converter (must change file path)
- Execute test_acp
- Evaluate with python absita_evaluation/evaluation_absita.py data/raw/absita_results_acp.csv absita_evaluation/absita_2018_test.csv 

If you don't want the acp part to be influenced by the acd part, then when you do test_acp, change file path and use the test.tsv file 

You could get some errors:
- Multiple names for shape: add positional argument shape= before shape tuple
- AttributeError: module 'tensorflow' has no attribute 'placeholder'
change tensorflow import with
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
- Empty figure: remove control
- No optimizers: use this optimizer tf.keras.optimizers.Optimizer
