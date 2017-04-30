import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from scipy.sparse import coo_matrix

cachedStopWords = stopwords.words("english")
# global vars here
word_dict = {}
word_freq = {}

tag_dict = {}
tag_freq = {}

train_data_frame = None
test_data_frame = None

col_array = []
val_array = []
row_array = []

#  global vars ends here

def readTestFile():
  file = './data/test.csv'
  data_frame = pd.read_csv(file, names=['id', 'title', 'content'])
  print('finished reading files ... ')
  data_frame['title'] = data_frame['title'].apply(lambda x : removestopword(x))
  data_frame['title'] = data_frame['title'].apply(lambda x : replace_special_character(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: removestopword(x))
  data_frame['content'] = data_frame['content'].apply(lambda x: replace_special_character(x))
  print('finished cleaning...')
  return data_frame

def replace_special_character(document):
    result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace('.', ' ')
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    result=re.sub(r'\b\w{1,3}\b', '', result)
    return result.strip()

def removestopword(document):
  text = ' '.join([word for word in document.strip().lower().split() if word not in cachedStopWords])
  return text

def readTrain(file):
  data_frame = pd.read_csv(file, names = ['id','title','content','tags'])
  print('finished reading files ... ')
  data_frame['title'] = data_frame['title'].apply(lambda x : x.decode('utf-8'))
  data_frame['content'] = data_frame['content'].apply(lambda x : x.decode('utf-8'))

  data_frame['text'] = data_frame[['title', 'content']].apply(lambda x : ''.join(x), axis = 1)

  data_frame['text'] = data_frame['text'].apply(lambda x : removestopword(x))
  data_frame['text'] = data_frame['text'].apply(lambda x : replace_special_character(x))

  # data_frame['content'] = data_frame['content'].apply(lambda x: removestopword(x))
  # data_frame['content'] = data_frame['content'].apply(lambda x: replace_special_character(x))
  data_frame['tags'] = data_frame['tags'].apply(lambda x : x.split(' '))
  data_frame.drop('title', 1, inplace= True)
  data_frame.drop('content', 1, inplace= True)
  print('finished cleaning...')
  return data_frame

def readTrainingDataSet():
  # files = ['./data/biology.csv','./data/cooking.csv','./data/crypto.csv','./data/diy.csv','./data/robotics.csv','./data/travel.csv']
  files = ['./data/cooking_short.csv','./data/crypto_short.csv']
  train_data_frames = []
  for f in files:
    train_data_frames.append(readTrain(f))
  return pd.concat(train_data_frames)

def assign_ids(document):
  doc = document.split()
  for w in doc:
    if w not in word_dict:  # assigning ID to the words
      word_dict[w] = len(word_dict)
    if w not in word_freq:  # creating the frequency count for the words
      word_freq[w] = 1
    else:
      word_freq[w] += 1
  return

def assign_tag_ids(document):
  for w in document:
    if w not in tag_dict:  # assigning ID to the words
      tag_dict[w] = len(tag_dict)
    if w not in tag_freq:  # creating the frequency count for the words
      tag_freq[w] = 1
    else:
      tag_freq[w] += 1
  return

def head_dict(dict, n):
  print('============ Dicitonary ============ ')
  for key in dict:
    if(n == 0):
      break
    print('{} , {}'.format(key, dict[key]))
    n -=1
  print('============ / Dicitonary ============ ')

def create_feature_ids(data_frame):
  data_frame['text'].apply(lambda x: assign_ids(x))
  data_frame['tags'].apply(lambda x: assign_tag_ids(x))

def create_coo_data(data_frame):
  row_id = 0;
  for index, row in data_frame.iterrows():
    doc = row['text'].split()
    doc_map = {}
    for w in doc:
      if w not in doc_map:
        doc_map[w] = 1
      else:
        doc_map[w] += 1
    for w in set(doc):
      col_array.append(word_dict[w])
      row_array.append(row_id)
      val_array.append(doc_map[w])
    row_id += 1
  return

def create_csr_matrix(coo_cols_array,coo_rows_array,coo_vals_array, rows, cols):
  csr_matrix = coo_matrix((coo_vals_array,(coo_rows_array, coo_cols_array)), shape = (rows, cols)).tocsr()
  return csr_matrix

def main():
  train_data_frame = readTrainingDataSet()
  print(train_data_frame.head())
  print('No of documents, features : {}'.format(train_data_frame.shape))
  create_feature_ids(train_data_frame)
  # print('Number of unique words : {}'.format(len(word_dict)))
  # print('=========================Sorted =========================')
  # print(sorted(word_freq.values(), reverse= True))
  # print('=========================Dict=========================')
  # print(sorted(word_dict.values(), reverse=True))

  # head_dict(tag_dict, 5)
  # head_dict(tag_freq, 5)

  create_coo_data(train_data_frame)


  coo_col_array = np.asarray(col_array)
  coo_row_array = np.asarray(row_array)
  coo_val_array = np.asarray(val_array)

  print('=========================Cols Array=========================')
  print('{}'.format(coo_col_array))

  print('=========================Row Array=========================')
  print('{}'.format(coo_row_array))

  print('=========================Value Array=========================')
  print('{}'.format(coo_val_array))

  totalRows = train_data_frame.shape[0]
  totalFeatures = len(word_dict)
  csr_matrix = create_csr_matrix(coo_col_array, coo_row_array, coo_val_array, totalRows, totalFeatures)

  print('=========================Csr Matrix=========================')
  print('{}'.format(csr_matrix))

  # # test = readTestFile()
  # print('=========================TEST=========================')
  # count = 0
  # for index, row in test.iterrows():
  #   if (count == 3):
  #     break
  #   print(row['title'])
  #   count += 1

if __name__ == '__main__':
  main()