import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy import Spotify

sp_oauth = SpotifyOAuth(client_id = 'd46e3db18c384e6ab33d369018d45fe0', client_secret = 'e8b047acdf984f0785e6f70ef58bb4df', redirect_uri = "https://stephen-t-2023.github.io/Spotify-Song-Recommendation/")

access_token = sp_oauth.get_cached_token()

sp = Spotify(auth_manager = sp_oauth)

playlists = sp.user_playlists("Mr_Cheesicus")

df = pd.read_csv("data.csv")

df.head()

feature_cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
                'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                'speechiness', 'tempo', 'time_signature', 'valence']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df = scaler.fit_transform(df[feature_cols])

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
