import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from datetime import datetime
import time
from sklearn.linear_model import SGDClassifier
from skmultilearn.problem_transform import BinaryRelevance
import sklearn.metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from skmultilearn.adapt import MLkNN
from nltk.tokenize import TreebankWordTokenizer


cachedStopWords = stopwords.words("english")
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\:)"  # :)
                           u"\:P"  # :p
                           u"\:D"  # :D
                           "]+", flags=re.UNICODE)

def remove_smileys(document):
  return emoji_pattern.sub(r'', document)

def replace_url(document):
  result = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", document).split())
  result = re.sub(r"http\S+", "", result)
  return result

def replace_specialchar(document):
  result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace(".", "")
  result = ' '.join(result.split())
  result = "".join(result.splitlines())
  return result.strip().lower()

def remove_stopwords(document):
  text = document.lower()
  text = ' '.join([word for word in text.split() if word not in cachedStopWords])
  temp = re.sub(r'\brt\b', '', text).strip(' ')
  temp = re.sub(r'\b\w{1,1}\b', '', temp)
  return temp

def remove_html_tags(document):
  p = re.compile(r'<.*?>')
  return re.sub(p, '', document)


def readTrain(file):
  print('reading file {}'.format(file))
  data_frame = pd.read_csv(file, names=['id', 'title', 'content', 'tags'], header=None)
  data_frame['title'] = data_frame['title'].apply(lambda x: remove_html_tags(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: remove_html_tags(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x: remove_html_tags(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x: x.split())

  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_smileys(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: replace_url(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))

  data_frame.drop('title', 1, inplace=True)
  data_frame.drop('content', 1, inplace=True)
  data_frame.drop('id', 1, inplace=True)
  data_frame.drop(data_frame.head(1).index, inplace=True)
  print('finished processing file : {}'.format(file))
  return data_frame

def readTrainingDataSet():
  files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv','./data/diy.csv','./data/robotics.csv','./data/travel.csv']
  # files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv']
  # files = ['./data/biology.csv']
  # files = ['./data/cooking_short.csv', './data/crypto_short.csv']
  # files = ['./data/cooking_short_train.csv']
  # files = ['./data/crypto_short.csv']
  # files = ['./data/tiny.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  data_frame = pd.concat(train_data_frames)
  data_frame.to_csv('concat.csv')
  return data_frame['text'].values, data_frame['tags'].values, files



def readPreProcessedData():
  data_frame = pd.read_csv('concat.csv', names=['text', 'tags'], header=None)
  print(data_frame.count())
  return data_frame['text'], data_frame['tags']


# def oneVsRest_BinRel_LogReg(X_train, X_test, Y_train, Y_test, word_dict, tags_dict, data_files):
#   print('Processing : oneVsRest_BinRel_LogReg')
#   print('-'*50)
#
#   Y_original = Y_test
#   # print('Oroginal Labels')
#   # print(Y_original)
#   # print('-' * 50)
#
#   vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
#   X_v_train = vectorizer.fit_transform(X_train)
#   X_v_test = vectorizer.fit_transform(X_test)
#   print('X_v_train shape : {}'.format(X_v_train.shape))
#   print('X_v_train {}'.format(X_v_test.toarray()))
#   print('-'*50)
#   print('X_v_test shape : {}'.format(X_v_test.shape))
#   print('X_v_test {}'.format(X_v_test.toarray()))
#   print('-'*50)
#
#
#   # No TFIDF
#   # transformer = TfidfTransformer(smooth_idf=False)
#   # X_train_tf = transformer.fit_transform(X_v_train)
#   # X_test_tf = transformer.fit_transform(X_v_test)
#
#   # print('X_vect_train shape : {}'.format(X_train_tf.shape))
#   # print('X_v_train {}'.format(X_train_tf.toarray()))
#   # print('-'*50)
#   # print('X_vect_test shape : {}'.format(X_test_tf.shape))
#   # print('X_v_test {}'.format(X_test_tf.toarray()))
#   # print('-'*50)
#   uniq_tags_names = list(tags_dict.keys())
#
#   # print('unique {}'.format(uniq_tags_names))
#   # print('-'*50)
#
#   mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
#   Y_train = mlb.fit_transform(Y_train)
#   Y_test = mlb.fit_transform(Y_test)
#
#   print('Y_train : {}'.format(Y_train))
#   print('Y_train {}'.format(Y_train))
#   print('-'*50)
#   print('Y_test {}'.format(Y_test))
#   print('-'*50)
#   # print('Y_train back : {}'.format(mlb.inverse_transform(Y_train)))
#   # print('-'*50)
#   # print('Y_test back : {}'.format(mlb.inverse_transform(Y_test)))
#   cls = LogisticRegression(C=40, class_weight='balanced')
#   classifier = BinaryRelevance(classifier=cls, require_dense=[False, True])
#
#   classifier.fit(X_v_train, Y_train)
#   Y_pred = classifier.predict(X_v_test)
#   score = sklearn.metrics.hamming_loss(Y_test, Y_pred)
#   print('-' * 50)
#   print('Score oneVsRest_BinRel_LogReg : {}'.format(score))
#   print('-' * 50)
#   return
#   Y_pred = classifier.predict(X_v_test)
#   # print('Pred labels')
#   # print('-' * 50)
#   Y_back = mlb.inverse_transform(Y_pred)
#   # print(Y_back)
#   # print('-' * 50)
#   write_to_file(Y_original, Y_back, 'oneVsRest_BinRel_SGDC_TfIdf', score, data_files)


#  a utility method to write the classification results to a file
import os
def write_to_file(Y_orig, Y_back, name, score, data_files):
  file_name = 'res_' + name +'_'
  file_name += str(datetime.now()).replace(' ','_')
  file_name += '.txt'
  f = open(os.path.join('/home/jayam/GIT/Transfer_Learning_Stack_Exchange/'+file_name),'w')
  f.write('Data Files : {}\n'.format(data_files))
  f.write('Score : {} \n'.format(score))
  f.write('\n\n')

  for i in range(len(Y_orig)):
    f.write('O  : {}\n'.format(Y_orig[i]))
    f.write('P  : {}\n'.format(Y_back[i]))
  f.close()
  print('~'*60)
  print('Completed: {} , written : {}'.format(name, file_name))
  print('~'*60)


def oneVsRest_SVCLinear(X_train, X_test, Y_train, Y_test, word_dict, tags_dict ):
  print('Processing : oneVsRest_LogReg')
  print('-'*50)

  Y_original = Y_test
  # print('Oroginal Labels')
  # print(Y_original)
  # print('-' * 50)

  vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)

  # print('X_vect_train shape : {}'.format(X_v_train.shape))
  # print('X_v_train {}'.format(X_v_train.toarray()))
  # print('-'*50)
  # print('X_vect_test shape : {}'.format(X_v_test.shape))
  # print('X_v_test {}'.format(X_v_test.toarray()))
  # print('-'*50)


  uniq_tags_names = list(tags_dict.keys())

  # print('unique {}'.format(uniq_tags_names))
  # print('-'*50)
  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  # print('Y_train : {}'.format(Y_train))
  Y_train = mlb.fit_transform(Y_train)
  Y_test = mlb.fit_transform(Y_test)
  # print('Y_train {}'.format(Y_train))
  # print('-'*50)
  # print('Y_test {}'.format(Y_test))
  # print('-'*50)
  # print('Y_train back : {}'.format(mlb.inverse_transform(Y_train)))
  # print('-'*50)
  # print('Y_test back : {}'.format(mlb.inverse_transform(Y_test)))

  classifier = OneVsRestClassifier(SVC(kernel='linear'))
  classifier.fit(X_v_train, Y_train)
  score = classifier.score(X_v_test, Y_test)
  print('-' * 50)
  print('Score oneVsRest_SVCLinear: {}'.format(score))
  print('-' * 50)
  Y_pred = classifier.predict(X_v_test)
  # print('Pred labels')
  # print('-' * 50)
  Y_back = mlb.inverse_transform(Y_pred)
  # print(Y_back)
  # print('-' * 50)
  write_to_file(Y_original, Y_back, 'oneVsRest_SVCLinear')

def oneVsRest_LogReg(X_train, X_test, Y_train, Y_test, word_dict, tags_dict ):
  print('Processing : oneVsRest_LogReg')
  print('-'*50)

  Y_original = Y_test
  # print('Oroginal Labels')
  # print(Y_original)
  # print('-' * 50)

  vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)

  # print('X_vect_train shape : {}'.format(X_v_train.shape))
  # print('X_v_train {}'.format(X_v_train.toarray()))
  # print('-'*50)
  # print('X_vect_test shape : {}'.format(X_v_test.shape))
  # print('X_v_test {}'.format(X_v_test.toarray()))
  # print('-'*50)

  uniq_tags_names = list(tags_dict.keys())

  # print('unique {}'.format(uniq_tags_names))
  # print('-'*50)
  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  # print('Y_train : {}'.format(Y_train))
  Y_train = mlb.fit_transform(Y_train)
  Y_test = mlb.fit_transform(Y_test)
  # print('Y_train {}'.format(Y_train))
  # print('-'*50)
  # print('Y_test {}'.format(Y_test))
  # print('-'*50)
  # print('Y_train back : {}'.format(mlb.inverse_transform(Y_train)))
  # print('-'*50)
  # print('Y_test back : {}'.format(mlb.inverse_transform(Y_test)))

  classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=0.01))
  classifier.fit(X_v_train, Y_train)
  score = classifier.score(X_v_test, Y_test)
  print('-' * 50)
  print('Score oneVsRest_LogReg : {}'.format(score))
  print('-' * 50)
  Y_pred = classifier.predict(X_v_test)
  # print('Pred labels')
  # print('-' * 50)
  Y_back = mlb.inverse_transform(Y_pred)
  # print(Y_back)
  # print('-' * 50)
  write_to_file(Y_original, Y_back, 'oneVsRest_LogReg')

def oneVsRest_LogReg_TfIdf(X_train, X_test, Y_train, Y_test, word_dict, tags_dict ):
  print('Processing : oneVsRest_LogReg_TfIdf')
  print('-'*50)

  Y_original = Y_test
  # print('Oroginal Labels')
  # print(Y_original)
  # print('-' * 50)

  vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)
  transformer = TfidfTransformer(smooth_idf=False)
  X_train_tf = transformer.fit_transform(X_v_train)
  X_test_tf = transformer.fit_transform(X_v_test)

  # print('X_vect_train shape : {}'.format(X_train_tf.shape))
  # print('X_v_train {}'.format(X_train_tf.toarray()))
  # print('-'*50)
  # print('X_vect_test shape : {}'.format(X_test_tf.shape))
  # print('X_v_test {}'.format(X_test_tf.toarray()))
  # print('-'*50)
  uniq_tags_names = list(tags_dict.keys())

  # print('unique {}'.format(uniq_tags_names))
  # print('-'*50)
  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  # print('Y_train : {}'.format(Y_train))
  Y_train = mlb.fit_transform(Y_train)
  Y_test = mlb.fit_transform(Y_test)
  # print('Y_train {}'.format(Y_train))
  # print('-'*50)
  # print('Y_test {}'.format(Y_test))
  # print('-'*50)
  # print('Y_train back : {}'.format(mlb.inverse_transform(Y_train)))
  # print('-'*50)
  # print('Y_test back : {}'.format(mlb.inverse_transform(Y_test)))

  classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=0.01))
  classifier.fit(X_train_tf, Y_train)
  score = classifier.score(X_test_tf, Y_test)
  print('-' * 50)
  print('Score oneVsRest_LogReg_TfIdf : {}'.format(score))
  print('-' * 50)
  Y_pred = classifier.predict(X_v_test)
  # print('Pred labels')
  # print('-' * 50)
  Y_back = mlb.inverse_transform(Y_pred)
  # print(Y_back)
  # print('-' * 50)
  write_to_file(Y_original, Y_back, 'oneVsRest_LogReg_TfIdf')

def oneVsRest_SGDC_TfIdf(X_train, X_test, Y_train, Y_test, word_dict, tags_dict, data_files):
  print('Processing : oneVsRest_SGDC_TfIdf')
  print('-'*50)

  Y_original = Y_test
  # print('Oroginal Labels')
  # print(Y_original)
  # print('-' * 50)

  vectorizer = CountVectorizer(min_df=1, vocabulary=word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)
  transformer = TfidfTransformer(smooth_idf=False)
  X_train_tf = transformer.fit_transform(X_v_train)
  X_test_tf = transformer.fit_transform(X_v_test)

  # print('X_vect_train shape : {}'.format(X_train_tf.shape))
  # print('X_v_train {}'.format(X_train_tf.toarray()))
  # print('-'*50)
  # print('X_vect_test shape : {}'.format(X_test_tf.shape))
  # print('X_v_test {}'.format(X_test_tf.toarray()))
  # print('-'*50)
  uniq_tags_names = list(tags_dict.keys())

  # print('unique {}'.format(uniq_tags_names))
  # print('-'*50)
  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  # print('Y_train : {}'.format(Y_train))
  Y_train = mlb.fit_transform(Y_train)
  Y_test = mlb.fit_transform(Y_test)
  # print('Y_train {}'.format(Y_train))
  # print('-'*50)
  # print('Y_test {}'.format(Y_test))
  # print('-'*50)
  # print('Y_train back : {}'.format(mlb.inverse_transform(Y_train)))
  # print('-'*50)
  # print('Y_test back : {}'.format(mlb.inverse_transform(Y_test)))

  # classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=0.01))
  classifier = OneVsRestClassifier(SGDClassifier(loss='hinge', alpha=1e-3,
                                     n_iter=50, learning_rate='optimal'))
  classifier.fit(X_train_tf, Y_train)
  score = classifier.score(X_test_tf, Y_test)
  print('-' * 50)
  print('Score oneVsRest_SGDC_TfIdf : {}'.format(score))
  print('-' * 50)
  Y_pred = classifier.predict(X_v_test)
  # print('Pred labels')
  # print('-' * 50)
  Y_back = mlb.inverse_transform(Y_pred)
  # print(Y_back)
  # print('-' * 50)
  write_to_file(Y_original, Y_back, 'oneVsRest_SGDC_TfIdf', score, data_files)




from sklearn import linear_model
from sklearn.metrics import accuracy_score

def oneVsRest_Pipeline(X_train, X_test, Y_train, Y_test,  tags_dict, words_dict, data_files):
  Y_original = Y_test
  print('Processing : oneVsRest_Pipeline_BINREL')
  print('-'*50)

  ######## for shapees
  vectorizer = CountVectorizer(min_df = 1, vocabulary = words_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)
  print('X_v_train.shape : {}'.format(X_v_train.shape))
  print('X_v_test.shape : {}'.format(X_v_test.shape))
  print('words chck : {}'.format(len(words_dict)))

  uniq_tags_names = list(tags_dict.keys())
  print('tags chck : {}'.format(len(uniq_tags_names)))

  # mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  mlb = preprocessing.MultiLabelBinarizer()
  Y_train = mlb.fit_transform(Y_train)
  print('Y_train.shape : {}'.format(Y_train.shape))
  Y_test = mlb.fit_transform(Y_test)
  print('Y_test.shape : {}'.format(Y_test.shape))
  print('-'*50)

  ########## for shapes ends


  # binRel_clf =
  # cls =
  #
  # clf = Pipeline([
  #   # ('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=TreebankWordTokenizer().tokenize)),
  #   ('vectorizer', CountVectorizer(tokenizer=TreebankWordTokenizer().tokenize)),
  #   # ('clf', cls),
  #   ('clf', BinaryRelevance(LinearSVC(), require_dense=[False,True])),
  # ])

  cls = linear_model.SGDClassifier(loss='hinge', alpha=1e-3,
                                   n_iter=50, random_state=None, learning_rate='optimal')

  # cv=CountVectorizer(ngram_range=(1,3),tokenizer=TreebankWordTokenizer().tokenize)

  # print cv.get_feature_names()
  # rf=RandomForestClassifier(n_estimators=1000)
  clf = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=TreebankWordTokenizer().tokenize)),
    ('clf', BinaryRelevance(classifier=cls, require_dense=[False, True]))])

  clf.fit(X_train, Y_train)

  Y_pred = clf.predict(X_test)
  print('Ypred shape :{}'.format(Y_pred.shape))
  print('Y val : \n{}'.format(Y_pred.toarray()))

  print(Y_pred[0])
  # res = ''.join(Y_pred[0].toarray())
  # print('res : {}'.format(res))
  Y_back = mlb.inverse_transform(Y_pred)
  print(Y_back)

  # temp = accuracy_score(y_test, predicted, normalize=True)
  # score = accuracy_score(Y_test, Y_pred, normalize=True)


  # score = classifier.score(X_test_tf, Y_test)
  score = 0
  print('-' * 50)
  print('Score oneVsRest_Pipeline_BINREL : {}'.format(score))

  # write_to_file(Y_original, None, 'oneVsRest_Pipeline_BINREL', score, data_files)


import  operator
def get_absolute_train_data(X, Y):
  words_dict = {}
  tags_dict = {}
  tags_freq = {}

  for x in X:
    for w in x.split():
      if w not in words_dict:
        words_dict[w] = len(words_dict)

  docId = 0
  for y in Y:
    for t in y:
      if t not in tags_dict:
        tags_dict[t] = len(tags_dict)
      if t not in tags_freq:
        tags_freq[t]  = []
      tags_freq[t].append(docId)
    docId += 1

  train_data_ids = []
  for k in tags_freq:
    train_data_ids.append(tags_freq[k][0])
    # print('tag : {} , occ docs  :{}'.format(k, len(tags_freq[k])))

  # threshold = 1
  # filtered_tags = {}
  # filtered_tags_doc_freq = {}
  # filtered_doc_ids = set()
  # for k in tags_freq:
  #   if(len(tags_freq[k]) > threshold):
  #     filtered_tags[k] = len(filtered_tags)
  #     if(k not in filtered_tags_doc_freq):
  #       filtered_tags_doc_freq[k] = set()
  #       for i in tags_freq[k]:
  #         filtered_tags_doc_freq[k].add(i)
  #     for i in tags_freq[k]:
  #       filtered_doc_ids.add(i)
  #
  # for k in filtered_tags_doc_freq:
  #   print('ft : {}, dfq : {}'.format(k, len(filtered_tags_doc_freq[k])))

  # print('abs_train shape : {}'.format(X_abs_train.shape))
  # print('filtered tags : {}'.format(len(filtered_tags)))
  # print('filtered docs : {}'.format(len(filtered_doc_ids)))

  # print('doc Ids : {}'.format(filtered_doc_ids))
  # filtered_doc_ids = list(filtered_doc_ids)
  # X_abs_train = X[filtered_doc_ids]
  # Y_abs_train = Y[filtered_doc_ids]

  X_abs_train = X[train_data_ids]
  Y_abs_train = Y[train_data_ids]

  return X_abs_train, Y_abs_train, tags_dict, words_dict

def main_two():
  # X, Y , data_files = readTrainingDataSet()
  X, Y = readPreProcessedData()


  print('Original shape X : {} Y: {}'.format(X.shape, Y.shape))
  # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
  # X_train , Y_train, tags_dict, words_dict = get_absolute_train_data(X_train, Y_train)
  # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

  X_s_train, Y_s_train, tags_dict, words_dict = get_absolute_train_data(X, Y)
  print('absolute  (after filtering) shape X : {} Y: {}'.format(X.shape, Y.shape))

  print('-'*50)
  # print(X_s_train)

  # print('-' * 50)
  # print(Y_s_train)

  for i in range(len(X_s_train)):
    print(X_s_train[i], Y_s_train[i])



  # X_train, X_test, Y_train, Y_test = train_test_split(X_s_train, Y_s_train, test_size=0.1, random_state=42)
  X_train, X_test, Y_train, Y_test = train_test_split(X_s_train, Y_s_train, test_size=0.3)

  print('-'*50)
  print('X_train shape : {}'.format(X_train.shape))
  print('Y_train shape : {}'.format(Y_train.shape))
  print('X_test shape : {}'.format(X_test.shape))
  print('Y_test shape : {}'.format(Y_test.shape))
  print('words count : {}'.format(len(words_dict)))
  print('tags count : {}'.format(len(tags_dict)))

  # word_dict = {}
  # for x in X_train:
  #   for w in x.split():
  #     if (w not in word_dict):
  #       word_dict[w] = len(word_dict)
  #
  # for x in X_test:
  #   for w in x.split():
  #     if (w not in word_dict):
  #       word_dict[w] = len(word_dict)

  # uniq_words_count = len(word_dict)
  # print('Unique words : {}'.format(uniq_words_count))

  # tags_dict = {}
  # for y in Y_train:
  #   for t in y:
  #     if (t not in tags_dict):
  #       tags_dict[t] = len(tags_dict)
  #
  # for y in Y_test:
  #   for t in y:
  #     if (t not in tags_dict):
  #       tags_dict[t] = len(tags_dict)

  uniq_tags_count = len(tags_dict)
  print('Unique tags : {}'.format(uniq_tags_count))
  print('-'*50)
  # start = timeit.timeit()
  # oneVsRest_LogReg(X_train, X_test, Y_train, Y_test, word_dict, tags_dict)
  # end = timeit.timeit()
  # print('Execution time : {}'.format(end-start))

  # start = timeit.timeit()
  # oneVsRest_SVCLinear(X_train, X_test, Y_train, Y_test, word_dict, tags_dict)
  # end = timeit.timeit()
  # print('Execution time : {}'.format(end - start))

  # start = timeit.timeit()
  # oneVsRest_LogReg_TfIdf(X_train, X_test, Y_train, Y_test, word_dict, tags_dict)
  # end = timeit.timeit()
  # print('Execution time : {}'.format(end-start))

  # start = time.time()
  # oneVsRest_SGDC_TfIdf(X_train, X_test, Y_train, Y_test, words_dict, tags_dict, data_files)
  # end = time.time()
  # print('Execution time : {} secs'.format(end-start))


  # start = time.time()
  # oneVsRest_BinRel_LogReg(X_train, X_test, Y_train, Y_test, word_dict, tags_dict, data_files)
  # end = time.time()
  # print('Execution time : {} secs'.format(end-start))

  oneVsRest_Pipeline(X_train, X_test, Y_train, Y_test, tags_dict, words_dict, data_files)
  # oneVsRest_Pipeline_Try(X_train, X_test, Y_train, Y_test, word_dict, tags_dict, data_files)

if __name__ == '__main__':
  main_two()
  # readTrainingDataSet()
  # readPreProcessedData()
