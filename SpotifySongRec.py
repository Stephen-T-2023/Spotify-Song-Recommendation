import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

df = pd.read_csv("data.csv")

df.head()

df.info()

feature_cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
                'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                'speechiness', 'tempo', 'time_signature', 'valence']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df = scaler.fit_transform(df[feature_cols])

print(normalised_df[:2])

indices = pd.Series(df.index, index = df['song_title']).drop_duplicates()

cosine = cosine_similarity(normalised_df)

def generate_recommendation(song_title, model_type = cosine):

    index = indices[song_title]
    score = list(enumerate(model_type[indices['Parallel Lines']]))

    similarity_score = sorted(score,key = lambda x:x[1], reverse = True)
    similarity_score = similarity_score[1:11]

    top_songs_index = [i[0] for i in similarity_score]

    top_songs = df['song_title'].iloc[top_songs_index]
    return top_songs

print("Recommended Songs using Cosine Similarity:")
print(generate_recommendation('Parallel Lines', cosine).values)

sig_kernel = sigmoid_kernel(normalised_df)

print("Recommended Songs using sig_kernel:")
print(generate_recommendation('Parallel Lines', sig_kernel).values)