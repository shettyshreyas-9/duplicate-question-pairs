import pandas
from sklearn.feature_extraction.text import CountVectorizer
import nltk



import os

print('Start')

# Replace '/path/to/your/directory' with the actual path where you want to store the NLTK data
custom_data_path = r'C:\shreyas\ML\campusx\projects\duplicate question pairs\nltk'


# Download 'punkt' manually
nltk.download('punkt', download_dir=custom_data_path)
nltk.download('stopwords', download_dir=custom_data_path)



# print(nltk.data.path.loc['C:\\Users\\shreyas\\AppData\\Roaming\\nltk_data'])
print('Done')