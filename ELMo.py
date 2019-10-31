import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub.estimator import LatestModuleExporter
from tensorflow_hub import tf_utils


pd.set_option('display.max_colwidth', 200)

train = pd.read_csv("train_tweet.csv")
test = pd.read_csv("test_tweet.csv")

train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
test['clean_tweet'] = test['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))

punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

train['clean_tweet'] = train['clean_tweet'].str.lower()
test['clean_tweet'] = test['clean_tweet'].str.lower()

train['clean_tweet'] = train['clean_tweet'].apply(lambda x:' '.join(x.split()))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output


train['clean_tweet'] = lemmatization(train['clean_tweet'])
test['clean_tweet'] = lemmatization(test['clean_tweet'])

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)


def elmo_vectors(x):
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))


list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]


elmo_train = [elmo_vectors(x['clean_tweet']) for x in list_train]
#elmo_test = [elmo_vectors(x['clean_tweet']) for x in list_test]

print(elmo_train)

'''''
elmo_train_new = np.concatenate(elmo_train, axis = 0)
elmo_test_new = np.concatenate(elmo_test, axis = 0)

pickle_out = open("elmo_train_03032019.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()

pickle_out = open("elmo_test_03032019.pickle","wb")
pickle.dump(elmo_test_new, pickle_out)
pickle_out.close()

pickle_in = open("elmo_train_03032019.pickle", "rb")
elmo_train_new = pickle.load(pickle_in)

pickle_in = open("elmo_test_03032019.pickle", "rb")
elmo_test_new = pickle.load(pickle_in)

from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, train['label'], random_state=42, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

lreg = LogisticRegression()
lreg.fit(xtrain, ytrain)

preds_valid = lreg.predict(xvalid)

f1_score(yvalid, preds_valid)

preds_test = lreg.predict(elmo_test_new)

sub = pd.DataFrame({'id':test['id'], 'label':preds_test})

sub.to_csv("sub_lreg.csv", index=False)
'''





