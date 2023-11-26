
import pandas
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pickle


# tp download the NLTK data in req. dir.
custom_data_path = r'C:\shreyas\ML\campusx\projects\duplicate question pairs\nltk'

nltk.download('punkt', download_dir=custom_data_path)
nltk.download('stopwords', download_dir=custom_data_path)

# for stopwords
from nltk.corpus import stopwords
# print(stopwords.words('english'))

# for punctuation
# import string
# print(string.punctuation)

# stemming
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()



import numpy as np
import math
from collections import Counter




lr= pickle.load(open('lr.pkl','rb'))

# Data Preprocessing: use the function created above for preprocessing text

def preprocess(text):
  # lower
  text= text.lower().strip()

  # tokenize
  text= nltk.word_tokenize(text)

  # only keep alphabets & numbers(this removes punctuation)
  a=[]
  for i in text:
    if i.isalnum():
      a.append(i)

  # remove stopwords
  b=[]
  for i in a:
    if i not in stopwords.words('english'):
      b.append(i)

  # stem the words to root form
  c=[]
  for i in b:
    c.append(ps.stem(i))

  return " ".join(c)



def query_point_creator(q1,q2):
  q1= preprocess(q1)
  q2= preprocess(q2)

  # Feature engineering (transformation) : BOW (vectorize the text)
    # merge texts
  questions= q1.split(' ')+ q2.split(' ')

  cv = CountVectorizer(max_features=3000)
  bow_array= cv.fit_transform(questions).toarray()
  q1_arr = bow_array[:round(math.ceil(len(bow_array)/2))]
  q2_arr = bow_array[round(math.ceil(len(bow_array)/2)):]

  if len(q1_arr)!=len(q2_arr):
    # Create a new row of zeros
    new_row = np.zeros((1, q2_arr.shape[1]),dtype=int)
    # Stack the existing array with the new row
    q2_arr_new = np.vstack((q2_arr, new_row))

  else:
    q2_arr_new= q2_arr


    # Horizontally stack the arrays
  q1_q1_array = np.hstack((q1_arr, q2_arr_new))

  # reshape the array to 1-D array
  q1_q1_reshaped= q1_q1_array.reshape(1,q1_q1_array.size)


  # adding 6000- remaining features for model pred
  rem_arr= np.zeros((1, 6000-len(q1_q1_reshaped[0])),dtype=int)
  q1_q2_6000= np.hstack((q1_q1_reshaped,rem_arr))




  # Feature engineering (addition)
  q1_len= len(q1)
  q2_len= len(q2)

  q1_word_num= len(q1.split(' '))
  q2_word_num= len(q2.split(' '))

    # common words
  w1 = set(q1.split(' '))
  w2 = set(q2.split(' '))
  wc= len(w1&w2)

    # total words
  w3 = set(q1.split(' '))
  w4 = set(q2.split(' '))
  wt= len(w3|w4)

    # word share
  ws= round(wc/wt,2)

  added_features= np.array((q1_len,q2_len,q1_word_num,q2_word_num,wc,wt,ws))
  added_features= added_features.reshape(1,added_features.shape[0])


  # final : BOW + Added features
  final_array= np.hstack((q1_q2_6000,added_features))


  # Model Training & Testing

  # pred= rf.predict(final_array)
  # pred2= xgbc.predict(final_array)
  pred3= lr.predict(final_array)
  # pred4= mnb.predict(final_array)

  # lst=[pred[0]]+[pred2[0]]+[pred3[0]]+[pred4[0]]

  # Count occurrences of each item
  # item_counts = Counter(lst)

  # Find the item with the maximum count
  # most_frequent_item = max(item_counts, key=item_counts.get)

  if pred3[0] == 0:
    return "Non Duplicate"
  else:
    return "Duplicate"

  # return (q1,q2)
  # return (pred,pred2,pred3,pred4)
  # return(most_frequent_item)


print('Done 1')

q1=' Will there really be any war between India and Pakistan over the Uri attack? What will be its effects?'
q2= 'Will there be a nuclear war between India and Pakistan?'
u=query_point_creator(q1,q2)
u


print(u)