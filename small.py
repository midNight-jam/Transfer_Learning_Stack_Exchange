import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


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

def readTrainingDataSet_Split_80_20():
  files = ['./data/tiny.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  data_frame = pd.concat(train_data_frames)
  # return data_frame['text'].values, data_frame['tags'].values
  total = data_frame.shape[0]
  percent_20 = int(total * 0.2)
  print('Total : {}   20 % {}'.format(total, percent_20))
  mask = np.random.rand((total)) < 0.8
  split_80 = data_frame[mask]
  split_20 = data_frame[~mask]
  print('split 80 Shape : {}'.format(split_80.shape))
  print('split 20 Shape : {}'.format(split_20.shape))
  return split_80['text'].values, split_80['tags'].values, split_20['text'].values, split_20['tags'].values

def readTrainingDataSet():
  files = ['./data/tiny.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  data_frame = pd.concat(train_data_frames)
  return data_frame['text'].values, data_frame['tags'].values

def main():
  # X_train, Y_train, X_test, Y_test = readTrainingDataSet()
  X, Y = readTrainingDataSet()
  print('Origianl X shape ; {}'.format(X.shape))
  print('Origianl Y shape ; {}'.format(Y.shape))
  X_train, X_test, Y_train,  Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


  print('X_train shape : {}'.format(X_train.shape))
  # print(X_train)
  print('Y_train shape : {}'.format(Y_train.shape))
  print(Y_train)
  print('Oroginal Labels')
  print(Y_test)
  word_dict = {}
  for x in X_train:
    for w in x.split():
      if(w not in word_dict):
        word_dict[w] = len(word_dict)

  for x in X_test:
    # print(x)
    # print('-'*20)
    for w in x.split():
      if (w not in word_dict):
        word_dict[w] = len(word_dict)

  uniq_words = len(word_dict)

  # print('Unique words : {}'.format(uniq_words))
  vectorizer = CountVectorizer(min_df=1, vocabulary = word_dict)
  X_v_train = vectorizer.fit_transform(X_train)
  X_v_test = vectorizer.fit_transform(X_test)

  # print('X_vect_train shape : {}'.format(X_v_train.shape))
  # print(X_v_train.toarray())
  # print('X_vect_test shape : {}'.format(X_v_test.shape))
  # print(X_v_test.toarray())

  tags_dict = {}
  for y in Y_train:
    # print(y)
    for t in y:
      if (t not in tags_dict ):
        tags_dict[t] = len(tags_dict )
  uniq_tags = len(tags_dict )
  # print('Unique tags : {}'.format(uniq_tags))

  uniq_tags_names = list(tags_dict.keys())
  # print(uniq_tags_names)
  mlb = preprocessing.MultiLabelBinarizer(classes=uniq_tags_names)
  # print(Y_train)
  Y_train = mlb.fit_transform(Y_train)

  classifier = OneVsRestClassifier(SVC(kernel='linear'))
  classifier.fit(X_v_train,Y_train)

  Y_pred = classifier.predict(X_v_test)
  # print(Y_pred)
  print('Pred labels')
  Y_back = mlb.inverse_transform(Y_pred)
  print(Y_back)


if __name__ == '__main__':
  main()
