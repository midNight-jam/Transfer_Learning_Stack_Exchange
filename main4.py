import numpy as np
import pandas as pd
import re
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from nltk.tokenize import TreebankWordTokenizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, hamming_loss, zero_one_loss
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

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

# def replace_hyphen(document):
#   return document.replace('-', '_')

def readTrain(file):
  print('reading file {}'.format(file))
  data_frame = pd.read_csv(file, names=['id', 'title', 'content', 'tags'], header=None)
  data_frame['title'] = data_frame['title'].apply(lambda x: remove_html_tags(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: remove_html_tags(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x: remove_html_tags(x))
  # data_frame['tags'] = data_frame['tags'].apply(lambda x: replace_hyphen(x))
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
  # files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv','./data/diy.csv','./data/robotics.csv','./data/travel.csv']
  # files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv']
  # files = ['./data/cooking_short.csv', './data/crypto_short.csv']
  files = ['./data/cooking_short_train.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  data_frame = pd.concat(train_data_frames)
  return data_frame['text'].values, data_frame['tags'].values

def readTestDataSet():
  file = './data/test_short.csv'
  # file = './data/test.csv'
  print('finished reading testfile : {}'.format(file))
  test_data_frame = pd.read_csv(file, names=['id', 'title', 'content','tags'])
  test_data_frame['title'] = test_data_frame['content'].apply(lambda x: remove_html_tags(x))
  test_data_frame['content'] = test_data_frame['content'].apply(lambda x: remove_html_tags(x))
  # test_data_frame['tags'] = test_data_frame['tags'].apply(lambda x: replace_hyphen(x))
  # test_data_frame['tags'] = test_data_frame['tags'].apply(lambda x: x.split())

  test_data_frame['text'] = test_data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
  test_data_frame['text'] = test_data_frame['text'].apply(lambda x: remove_smileys(x))
  test_data_frame['text'] = test_data_frame['text'].apply(lambda x: replace_url(x))
  test_data_frame['text'] = test_data_frame['text'].apply(lambda x: remove_stopwords(x))
  test_data_frame.drop('title', 1, inplace=True)
  test_data_frame.drop('content', 1, inplace=True)
  test_data_frame.drop('id', 1, inplace=True)
  test_data_frame.drop(test_data_frame.head(1).index, inplace=True)
  print('finished cleaning Test data')
  return test_data_frame['text'].values

def convert_to_dict(data_frame, dict):
  for t in data_frame:
    for w in t:
      if w not in dict:
        dict[w] = len(dict)
  print('len : {}'.format(len(dict)))

def head_dict(dict, n, title):
  print('============ {} ============ '.format(title))
  for key in dict:
    if (n == 0):
      break
    print('{} , {}'.format(key, dict[key]))
    n -= 1
  print('============ /{}/ ============ '.format(title))

def print_predicted_labels_one(predicted, target_classes):
  dd = predicted
  row_id = 0
  for d in dd:
    index = 0
    # labels = []
    # for i in d:
    #   if (i == 1):
    #     labels.append(target_classes[index])
    #   index += 1
    # print('doc : {}   Lables : {}'.format(row_id,labels))
    print(d)
    row_id += 1

def print_predicted_labels(predicted, target_classes):
  dd = predicted
  row_id = 0
  for d in dd:
    index = 0
    labels = []
    for i in d:
      if (i == 1):
        labels.append(target_classes[index])
      index += 1
    print('doc : {}   Lables : {}'.format(row_id,labels))
    row_id += 1

def write_predicted_labels(file_name, predicted, target_classes):
  f = open(file,'w')
  for p in predicted:
    index = 0
    labels = []
    print(p)
    # for i in p:
    #   print()
      # if (i == 1):
      #   labels.append(target_classes[index])
      # index += 1
    str = ''.join(labels)
    print(type(str))
    # f.write(str+ '\n')
  # f.close()

def classify_labels(train_X, train_Y, test_X, test_Y, target_classes):
  mlb = preprocessing.MultiLabelBinarizer(classes=target_classes)
  y_train = mlb.fit_transform(train_Y)
  x_train, y_train = shuffle(train_X, y_train)
  x_test = np.array(test_X)

  mlb = preprocessing.MultiLabelBinarizer(classes=target_classes)
  y_true = mlb.fit_transform(test_Y)

  cls = linear_model.SGDClassifier(loss='hinge', alpha=1e-3,
                                   n_iter=500, random_state=None, learning_rate='optimal')
  classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=TreebankWordTokenizer().tokenize)),
    ('clf', BinaryRelevance(classifier=cls, require_dense=[False, True]))])
  print('X_train_shape :  {}'.format(x_train.shape))
  print('y_train_shape :  {}'.format(y_train.shape))
  # print(x_train)
  print(y_train)
  # for d in y_train:
  #   print(d)
  classifier.fit(x_train, y_train)
  # y_pred = classifier.predict(x_test).toarray()
  # print(y_pred)
  # print('y_true shape : {} '.format(y_true.shape))
  # print('y_pred shape : {} '.format(y_pred.shape))
  # score = accuracy_score(y_true, y_pred)
  # norm_score = accuracy_score(y_true, y_pred, normalize= True)
  # print('')
  # print('Score : {} '.format(score))
  # print('Normalized Score : {} '.format(norm_score))
  # print('')
  # print_predicted_labels(y_pred, target_classes)

def classify_labels_two(train_X, train_Y, test_X, target_classes):
  mlb = preprocessing.MultiLabelBinarizer(classes=target_classes)
  x_train = np.array(train_X)
  y_train = mlb.fit_transform(train_Y)
  x_test = np.array(test_X)

  # mlb = preprocessing.MultiLabelBinarizer(classes= true_test_target_classes)
  # y_true = mlb.fit_transform(test_Y)


  classifier = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1, max_df=2)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))
  ])

  classifier.fit(x_train, y_train)
  y_pred = classifier.predict(x_test)
  print(y_pred)
  # score = accuracy_score(y_true, y_pred)
  # norm_score = accuracy_score(y_true, y_pred, normalize= True)
  # print('')
  # print('Score : {} '.format(score))
  # print('Normalized Score : {} '.format(norm_score))
  # print('')
  print_predicted_labels(y_pred, target_classes)
  # write_predicted_labels('res_4.txt', y_pred, target_classes)

  # kf = ShuffleSplit(n_splits=5, random_state=0)
  # for train_index, test_index in kf.split(x_train, y_train):
  #   classifier.fit(x_train[train_index], y_train[train_index])
  #   predicted = classifier.predict(x_test)
  #   print(predicted.shape)
  #   print(predicted)

def main():
  train_X, train_Y = readTrainingDataSet()
  test_X = readTestDataSet()
  tags_dict = {}

  convert_to_dict(train_Y, tags_dict)
  target_names = []
  for key in tags_dict.iterkeys():
    target_names.append(key)

  # print('Train target clasess')
  # print(target_names)
  # print('------------------------------------------')
  # print('Apporach 1  Pipeline : Count Vectorizer +  binaryrelevAnce with SGD Classifier')
  # classify_labels(train_X, train_Y, test_X, test_Y, target_names)
  print('Apporach 2 Pipeline : Count Vectorizer +  TfIdf + OneVsRestClassifier with LinearSVC')
  classify_labels_two(train_X, train_Y, test_X, target_names)
  # print('Apporach 3 Pipeline : Count Vectorizer +  TfIdf + OneVsRestClassifier with LinearSVC')
  # classify_labels_two(train_X, train_Y, test_X, target_names)

if __name__ == '__main__':
  main()
