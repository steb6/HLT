from keras.layers import Dropout, Dense, Bidirectional, LSTM, Embedding, GaussianNoise, concatenate, RepeatVector
import tensorflow as tf
from keras.engine import Input
from keras.regularizers import l2
from kutilities.layers import AttentionWithContext, MeanOverTime
from keras.engine import Model
from keras.optimizers import Adam

# drop_rep is dropout of representation, drop_out is dropout after final dense
def model(wv, tweet_max_length, aspect_max_length, classes, task, rnn_cells=0, final_cells=0, drop_rep=0, drop_out=0):
    ######################################################
    # HyperParameters
    ######################################################
    # noise = kwargs.get("noise", 0)
    # trainable = kwargs.get("trainable", False)
    # final_size = kwargs.get("final_size", 100)
    # final_type = kwargs.get("final_type", "linear")
    # drop_text_input = kwargs.get("drop_text_input", 0.)
    # drop_text_rnn = kwargs.get("drop_text_rnn", 0.)
    # drop_text_rnn_U = kwargs.get("drop_text_rnn_U", 0.)
    # drop_target_rnn = kwargs.get("drop_target_rnn", 0.)
    # drop_rep = kwargs.get("drop_rep", 0.)
    # drop_final = kwargs.get("drop_final", 0.)
    # activity_l2 = kwargs.get("activity_l2", 0.)
    # clipnorm = kwargs.get("clipnorm", 5)
    # lr = kwargs.get("lr", 0.001)

    ######################################################
    # NEW HyperParameters
    ######################################################
    drop_rnn = 0.2
    noise = 0.2
    drop_text_rnn = 0.3
    drop_target_rnn = 0.2
    drop_text_input = 0.3

    drop_rep = 0.2
    drop_out = 0.3

    final_type = "linear"
    activity_l2 = 0.001
    clipnorm = .1
    lr = 0.001

    #####################################################
    # shared_RNN = Bidirectional(LSTM(75, return_sequences=True, consume_less='cpu', dropout_U=drop_text_rnn_U,
    # W_regularizer=l2(0)))
    shared_RNN = Bidirectional(LSTM(rnn_cells, dropout=drop_rnn, return_sequences=True))

    # GET the right model, if wv is None, then we already have the embeddings (BERT)
    if wv is not None:
        input_tweet = Input(shape=[tweet_max_length], dtype='int32')
        input_aspect = Input(shape=[aspect_max_length], dtype='int32')
        tweets_emb = Embedding(input_dim=wv.shape[0],
                               output_dim=wv.shape[1],
                               input_length=tweet_max_length,
                               trainable=False,
                               mask_zero=True,
                               weights=[wv])(input_tweet)
        tweets_emb = GaussianNoise(noise)(tweets_emb)
        tweets_emb = Dropout(drop_text_input)(tweets_emb)

        aspects_emb = Embedding(input_dim=wv.shape[0],
                                output_dim=wv.shape[1],
                                input_length=aspect_max_length,
                                trainable=False,
                                mask_zero=True,
                                weights=[wv])(input_aspect)
        aspects_emb = GaussianNoise(noise)(aspects_emb)
        h_tweets = shared_RNN(tweets_emb)
        h_tweets = Dropout(drop_text_rnn)(h_tweets)

        h_aspects = shared_RNN(aspects_emb)
        h_aspects = Dropout(drop_target_rnn)(h_aspects)
        h_aspects = MeanOverTime()(h_aspects)
        h_aspects = RepeatVector(tweet_max_length)(h_aspects)
    else:
        # We give already the embedding
        input_tweet = Input(shape=(tweet_max_length, 768,), dtype='float32')
        input_aspect = Input(shape=(aspect_max_length, 768,), dtype='float32')
        h_tweets = shared_RNN(input_tweet)
        h_tweets = Dropout(drop_text_rnn)(h_tweets)

        h_aspects = shared_RNN(input_aspect)
        h_aspects = Dropout(drop_target_rnn)(h_aspects)
        h_aspects = MeanOverTime()(h_aspects)
        h_aspects = RepeatVector(tweet_max_length)(h_aspects)

    # Merge of Aspect + Tweet
    if task == "acp":
        representation = concatenate([h_tweets, h_aspects])
    else:
        representation = h_tweets

    # apply attention over the hidden outputs of the RNN's
    representation = AttentionWithContext()(representation)
    representation = Dropout(drop_rep)(representation)
    # if task == "acp":
    #    representation = AttentionWithContext()(representation)
    #    representation = Dropout(drop_rep)(representation)
    # else:
    #    representation = Attention()(representation)

    # Default is linear, should try maxout
    # if final_type == "maxout":
    #   representation = MaxoutDense(final_size)(representation)
    # else:
    representation = Dense(final_cells, activation=final_type)(
            representation)
    representation = Dropout(drop_out)(representation)

    ######################################################
    # Probabilities
    ######################################################

    probabilities = Dense(classes, activation="softmax" if task is "acp" else "sigmoid",
                          activity_regularizer=l2(activity_l2))(representation)

    final_model = Model(inputs=[input_aspect, input_tweet] if task is "acp" else input_tweet,
                        outputs=probabilities)

    final_model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                        loss="categorical_crossentropy" if task is "acp" else "binary_crossentropy",
                        metrics=['accuracy', tf.keras.metrics.Recall()])

    return final_model
