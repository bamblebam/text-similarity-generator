# %%
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')  # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


'''remove punctuation, lowercase, stem'''


def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]


# %%
a = 'Machine learning is the study of computer algorithms that improve automatically through experience.\
Machine learning algorithms build a mathematical model based on sample data, known as training data.\
The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
where no fully satisfactory algorithm is available.'
b = 'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.'
print(cosine_sim(a, b))

# %%
