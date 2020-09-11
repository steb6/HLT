To download old embeddings (if it will be online again one day): http://hlt.isti.cnr.it/wordembeddings/
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

To check learning results, you can use tensorboard by just writing
- tensorboard --logdir logs / [alberto / w2v] / [acd / acp]