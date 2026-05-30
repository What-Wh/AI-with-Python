# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Завантаження необхідних мовних ресурсів
nltk.download("punkt")  # Для токенізації
nltk.download("punkt_tab")  # Для токенізації речень
nltk.download("stopwords")  # Стоп-слова
nltk.download("wordnet")  # Для лемматизації
nltk.download("omw-1.4")  # WordNet мовні дані

warnings.filterwarnings(action="ignore")


#  Reads ‘alice.txt’ file
sample = open("./feedback.txt", "r", encoding="utf-8")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# --- 2. Стоп-слова ---
print("\n=== Видалення стоп-слів ===")
stop_words = set(stopwords.words("english"))
filtered_words = [
    word for sentence in data for word in sentence if word.lower() not in stop_words and word.isalpha()
]

# Create CBOW model
model1 = gensim.models.Word2Vec([filtered_words], min_count=1, vector_size=100, window=5, sg=1)

f = model1.wv.most_similar("well", topn=3)
print("Word: well; Similarity: ", f[0][0], " ", f[1][0], " ", f[2][0], "; Rate: ", f[0][1], " ", f[1][1], " ", f[2][1])