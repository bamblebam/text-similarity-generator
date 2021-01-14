# %%
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# %%
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


# %%
a = 'Machine learning is the study of computer algorithms that improve automatically through experience.\
Machine learning algorithms build a mathematical model based on sample data, known as training data.\
The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
where no fully satisfactory algorithm is available.'
b = 'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.'
c = "Software development is for dumb dumbs"
sentences = [a, c]
# %%
sentence_embeddings = sbert_model.encode(sentences)
# %%
cos_sim = cosine_similarity(sentence_embeddings)
# %%
print(cos_sim)

# %%
