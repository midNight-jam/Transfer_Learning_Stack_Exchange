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
                           u"\:)"  #  :)
                           u"\:P"  # :p
                           u"\:D"  # :D
                           "]+", flags=re.UNICODE)

def remove_smileys(document):
  return emoji_pattern.sub(r'',document)

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

def replace_hyphen(document):
  return document.replace('-','_')

def readTrain(file):
  print('reading file {}'.format(file))
  data_frame = pd.read_csv(file, names = ['id','title','content','tags'], header=None)
  data_frame['title'] = data_frame['title'].apply(lambda x : remove_html_tags(x))
  data_frame['content'] = data_frame['content'].apply(lambda x : remove_html_tags(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x : remove_html_tags(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x : replace_hyphen(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x : x.split())

  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x : ''.join(x), axis = 1)
  data_frame['text'] = data_frame['text'].apply(lambda x : remove_smileys(x))
  data_frame['text'] = data_frame['text'].apply(lambda x : replace_url(x))
  data_frame['text'] = data_frame['text'].apply(lambda x : remove_stopwords(x))

  data_frame.drop('title', 1, inplace= True)
  data_frame.drop('content', 1, inplace= True)
  data_frame.drop('id', 1, inplace= True)
  data_frame.drop(data_frame.head(1).index, inplace=True)
  print('finished processing file : {}'.format(file))
  return data_frame

def readTrainingDataSet():
  # files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv','./data/diy.csv','./data/robotics.csv','./data/travel.csv']
  files = ['./data/cooking_short.csv','./data/crypto_short.csv']
  # files = ['./data/cooking_short.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  data_frame = pd.concat(train_data_frames)
  # print('--------------------- head --------------------- ')
  # print(data_frame.head)
  # print('--------------------- / head / --------------------- ')
  return data_frame['text'].values, data_frame['tags'].values
  # vectorizer = CountVectorizer(min_df=1, max_features=155190)
  # X = vectorizer.fit_transform(data_frame['text'].values)
  # transformer = TfidfTransformer(smooth_idf=False)
  # tfidf_text = transformer.fit_transform(X)
  # Y = vectorizer.fit_transform(data_frame['tags'].values)
  # tfidf_tags = transformer.fit_transform(Y)
  # return tfidf_text, tfidf_tags

def readTestDataSet():
  # file = './data/test.csv'
  file ='./data/cooking_short.csv'
  data_frame = pd.read_csv(file, names=['id', 'title', 'content'])
  print('finished reading files ... ')
  data_frame = pd.read_csv(file, names=['id', 'title', 'content'])
  data_frame['content'] = data_frame['content'].apply(lambda x: remove_html_tags(x))
  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_smileys(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: replace_url(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))
  data_frame.drop('title', 1, inplace=True)
  data_frame.drop('content', 1, inplace=True)
  data_frame.drop('id', 1, inplace=True)
  data_frame.drop(data_frame.head(1).index, inplace=True)
  print('finished cleaning Test data')
  return data_frame['text'].values

tags_dict = {}
def convert_to_dict(data_frame):
  # print(data_frame)
  for t in data_frame:
    for w in t:
      if w not in tags_dict:
        tags_dict[w] = len(tags_dict)
  print('len : {}'.format(len(tags_dict)))
  # print('{}'.format(tags_dict))

def head_dict(dict, n, title):
  print('============ {} ============ '.format(title))
  for key in dict:
    if(n == 0):
      break
    print('{} , {}'.format(key, dict[key]))
    n -=1
  print('============ /{}/ ============ '.format(title))

def classify_labels(train_X, train_Y, test_X, target_classes):
  mlb = preprocessing.MultiLabelBinarizer(classes=target_classes)
  y_train = mlb.fit_transform(train_Y)
  x_train, y_train = shuffle(train_X, y_train)
  x_test = np.array(test_X)
  # rows = y_train.shape[0]
  # y_test_text = []
  # for i in range(rows):
  #   y_test_text.append(['baking'])

  # y_test = mlb.fit_transform(y_test_text)
  # print(y_test)

  kf = ShuffleSplit(n_splits=5, random_state=0)
  cls = linear_model.SGDClassifier(loss='hinge', alpha=1e-3,
                                   n_iter=500, random_state=None, learning_rate='optimal')
  classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=TreebankWordTokenizer().tokenize)),
    ('clf', BinaryRelevance(classifier=cls, require_dense=[False, True]))])
  classifier.fit(x_train, y_train)
  predicted = classifier.predict(x_test)
  print(predicted.toarray())

def classify_labels_two(train_X, train_Y, test_X, target_classes):
  mlb = preprocessing.MultiLabelBinarizer(classes=target_classes)
  # mlb = preprocessing.MultiLabelBinarizer()
  x_train = np.array(train_X)
  y_train = mlb.fit_transform(train_Y)
  # y_train = np.array(train_Y)
  x_test = np.array(test_X)
  y_test_text = []
  # for t in range(len(train_Y)):
  for t in range(len(tags_dict)):
    arr = [random.choice(target_classes)]
    y_test_text.append(arr)
  # print(y_test_text)
  # print(len(y_test_text))
  y_test = mlb.fit_transform(y_test_text)

  # print(y_test)

  classifier = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1, max_df= 2)),
    ('tfidf', TfidfTransformer()),
    ('clf',OneVsRestClassifier(LinearSVC()))
  ])

  classifier.fit(x_train, y_train)
  predicted = classifier.predict(x_test)
  # print(predicted)
  # dd = predicted
  # for d in dd:
  #   index = 0
  #   for i in d:
  #     if(i ==1):
  #       print(target_classes[index])
  #     index +=1
  #   print('===============label==============')
  predicted_final = None
  score = 0.0
  kf = ShuffleSplit(n_splits=5, random_state=0)
  for train_index, test_index in kf.split(x_train, y_train):
    # print("TRAIN:", train_index)
    classifier.fit(x_train[train_index], y_train[train_index])
    # print "Training completed"
    predicted = classifier.predict(x_test)
    print(predicted.shape)
    print(y_test.shape)
    temp = predicted.toarray()
    temp = accuracy_score(y_test, predicted, normalize=True)
    # print (temp)
    # if (score < temp):
    #   score = temp
    #   print ('Accuracy:', score)
    #   predicted_final = predicted
  # temp = predicted_final.toarray()
  # print(temp)
  # print(predicted.shape)
  # print(y_test.shape)
  # score = accuracy_score(y_test, predicted)
  # print('Score : {}'.format(score))

  # print(classification_report(y_train, predicted, target_names=target_classes))
  # for item , lables in zip(x_test, predicted):
  #   print(' {} --> {}'.format(item, ','.join(target_classes[x] for x in lables)))

def main():
  train_X, train_Y = readTrainingDataSet()
  test_X = readTestDataSet()
  # print(train_Y.shape)
  # print(train_Y)
  # print('===================================================')
  # mlb = preprocessing.MultiLabelBinarizer()
  # y_train = mlb.fit_transform(train_Y)
  # print(list(mlb.classes_))
  # print(y_train)
  convert_to_dict(train_Y)
  target_names = []
  for key in sorted(tags_dict.iterkeys()):
    target_names.append(key)
  # print(train_classes)
  print('Apporach 1  Pipeline : Count Vectorizer +  binaryrelevAnce with SGD Classifier')
  classify_labels(train_X, train_Y, test_X, target_names)
  print('Apporach 2 Pipeline : Count Vectorizer +  TfIdf + OneVsRestClassifier with LinearSVC')
  classify_labels_two(train_X, train_Y, test_X, target_names)
  print('Apporach 3 Pipeline : Count Vectorizer +  TfIdf + OneVsRestClassifier with LinearSVC')
  classify_labels_two(train_X, train_Y, test_X, target_names)
  # print(train_X)
  # print(train_Y)
  # print(test_X)

if __name__ == '__main__':
  main()

  # unique words for complete data after preprocessing  : 155190
  # total no of documents in the complete data : 87006
