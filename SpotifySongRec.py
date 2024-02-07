import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

from rich.progress import Progress
from time import sleep

import threading

flag = True

#load credentials
load_dotenv("credentials.env")

#retrieve credentials from the env file
CLIENT_ID = os.getenv("CLIENT_ID", "")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "")
OUTPUT_FILE_NAME = "track_info.csv"

#the playlist link
print("Please choose one of the following options:")
print("1. Enter your own playlist link")
print("2. Use Spotifiy's own Best of the Decade playlist")
choice = input("")
while flag:
    if choice == "1":
        PLAYLIST_LINK = input("Enter the spotify playlist ID: ")
        flag = False
    elif choice == "2":
        PLAYLIST_LINK = "https://open.spotify.com/playlist/37i9dQZF1DWXADZ9KRTmmm?si=f4b952af45414ef1"
        flag = False
    else:
        choice = input("Please enter either 1 or 2:")

with Progress() as progress:
    processing = progress.add_task("[green]Processing Data...", total=1000)

    def process_data():
        progress.update(processing, advance = 200)

    def recom():
        client_credentials_manager = SpotifyClientCredentials(
            client_id=CLIENT_ID, client_secret=CLIENT_SECRET
        )

        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        # Function to get audio features for a given track ID
        def get_audio_features(track_id):
            audio_features = sp.audio_features(track_id)[0]
            return audio_features

        # Function to get track information (title, artist, target) for a given track ID
        def get_track_info(track_id):
            track_info = sp.track(track_id)
            title = track_info['name']
            artist = track_info['artists'][0]['name']
            target = 1  # Set your target value here if applicable
            return title, artist, target

        # Function to get all track IDs from a Spotify playlist
        def get_playlist_track_ids(playlist_id):
            results = sp.playlist_tracks(playlist_id)
            track_ids = [item['track']['id'] for item in results['items'] if item['track']]
            return track_ids

        # Input your Spotify playlist URL or ID
        playlist_url_or_id = PLAYLIST_LINK

        # Extract playlist ID from URL if provided
        if 'playlist/' in playlist_url_or_id:
            playlist_id = playlist_url_or_id.split('playlist/')[-1].split('?')[0]
        else:
            playlist_id = playlist_url_or_id

        process_data()

        # Get track IDs from the playlist
        track_ids = get_playlist_track_ids(playlist_id)

        # Header for the CSV file
        header = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", 
                "key", "liveness", "loudness", "mode", "speechiness", "tempo", 
                "time_signature", "valence", "target", "song_title", "artist"]

        # Open CSV file for writing
        with open('track_info.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write the header to the CSV file
            csv_writer.writerow(header)
            
            process_data()

            # Iterate through each track ID
            for track_id in track_ids:
                # Get audio features
                audio_features = get_audio_features(track_id)

                # Get track information
                title, artist, target = get_track_info(track_id)

                # Write the data to the CSV file
                csv_writer.writerow([audio_features[feature] for feature in header[:-3]] + [target, title, artist])
                
        df = pd.read_csv("track_info.csv")

        process_data()

        df.head()

        feature_cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
                        'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
                        'speechiness', 'tempo', 'time_signature', 'valence']

        process_data()

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalised_df = scaler.fit_transform(df[feature_cols])

        indices = pd.Series(df.index, index = df['song_title']).drop_duplicates()

        cosine = cosine_similarity(normalised_df)

        def generate_recommendation(song_title, model_type = cosine):

            index = indices[song_title]
            score = list(enumerate(model_type[index]))

            similarity_score = sorted(score,key = lambda x:x[1], reverse = True)
            similarity_score = similarity_score[1:11]

            top_songs_index = [i[0] for i in similarity_score]

            top_songs = df['song_title'].iloc[top_songs_index]
            return top_songs

        process_data()

        sleep(3)

        # Specify the CSV file path and the column index you want to print
        csv_file_path = 'track_info.csv'
        column_index_to_print = 14  # Change this to the index of the column you want (0-indexed)

        # Open the CSV file and read its content
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Iterate through each row and print the value of the specified column
            for row in csv_reader:
                if len(row) > column_index_to_print:
                    print(row[column_index_to_print])

        ost_choice = input("Choose one of these songs and we will find other songs in the playlist that are similar: ", )

        print("Recommended Songs using Cosine Similarity:")
        print(generate_recommendation(ost_choice , cosine).values)

        sig_kernel = sigmoid_kernel(normalised_df)

        print("Recommended Songs using sig_kernel:")
        print(generate_recommendation(ost_choice , sig_kernel).values)

        os.remove("track_info.csv")

    recom()