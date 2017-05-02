import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from nltk.tokenize import TreebankWordTokenizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score

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

tag_dict = {}
def assign_ids(document):
  doc = document.split()
  for w in doc:
    if w not in tag_dict:  # assigning ID to the words
      tag_dict[w] = len(tag_dict)
  return document

def readTrain(file):
  print('reading file {}'.format(file))
  data_frame = pd.read_csv(file, names = ['id','title','content','tags'])
  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x : ''.join(x), axis = 1)
  data_frame['text'] = data_frame['text'].apply(lambda x : remove_smileys(x))
  data_frame['text'] = data_frame['text'].apply(lambda x : replace_url(x))
  data_frame['text'] = data_frame['text'].apply(lambda x : remove_stopwords(x))

  # data_frame['text'] = data_frame['text'].apply(lambda x : replace_special_character(x))
  # data_frame['content'] = data_frame['content'].apply(lambda x: removestopword(x))
  # data_frame['content'] = data_frame['content'].apply(lambda x: replace_special_character(x))
  # data_frame['tags'] = data_frame['tags'].apply(lambda x : x.split(' '))
  data_frame.drop('title', 1, inplace= True)
  data_frame.drop('content', 1, inplace= True)
  data_frame.drop('id', 1, inplace= True)
  print('finished processing file : {}'.format(file))
  return data_frame

def readTrainingDataSet():
  # files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv','./data/diy.csv','./data/robotics.csv','./data/travel.csv']
  # files = ['./data/cooking_short.csv','./data/crypto_short.csv']
  files = ['./data/cooking_short.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  print('Data inside Dataframe for files {}'.format(files))
  data_frame = pd.concat(train_data_frames)

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
  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x: ''.join(x), axis=1)
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_smileys(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: replace_url(x))
  data_frame['text'] = data_frame['text'].apply(lambda x: remove_stopwords(x))
  data_frame.drop('title', 1, inplace=True)
  data_frame.drop('content', 1, inplace=True)
  data_frame.drop('id', 1, inplace=True)
  print('finished cleaning Test data')
  return data_frame['text'].values

def head_dict(dict, n, title):
  print('============ {} ============ '.format(title))
  for key in dict:
    if(n == 0):
      break
    print('{} , {}'.format(key, dict[key]))
    n -=1
  print('============ /{}/ ============ '.format(title))

def classify_labels(train_X, train_Y, test_X):
  y_train = MultiLabelBinarizer().fit_transform(train_Y)
  x_train, y_train = shuffle(train_X, y_train)
  x_test = np.array(test_X)
  kf = ShuffleSplit(n_splits=5, random_state=0)
  cls = linear_model.SGDClassifier(loss='hinge', alpha=1e-3,
                                   n_iter=500, random_state=None, learning_rate='optimal')
  classifier = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=TreebankWordTokenizer().tokenize)),
    ('clf', BinaryRelevance(classifier=cls, require_dense=[False, True]))])
  predicted_final = None
  score = 0.0
  # print(classifier)
  classifier.fit(x_train, y_train)
  predicted_final  = classifier.predict(x_test)
  print('predicted final {}'.format(predicted_final))
  print(predicted_final)
  for train_index, test_index in kf.split(x_train, y_train):
    print(train_index)
    print(test_index)
    classifier.fit(x_train[train_index], y_train[train_index])
    predicted = classifier.predict(x_test)
    print(predicted)
    current_score = accuracy_score(x_test, predicted, normalize= True)
    if(current_score > score):
      score = current_score
      print('Accuracy : {}',format(score))
      predicted_final = predicted
    temp = predicted_final
    print(temp)


  # print(classifier)
  # print('Y_train shape {}'.format(y_train.shape))
  # print(train_X)
  # print('X_train shape {}'.format(train_X.shape))

def main():
  train_X, train_Y = readTrainingDataSet()
  test_X = readTestDataSet()
  classify_labels(train_X, train_Y, test_X)
  # print(train_X)
  # print(train_Y)
  # print(test_X)

if __name__ == '__main__':
  main()
  # unique words for complete data after preprocessing  : 155190
  # total no of documents in the complete data : 87006
