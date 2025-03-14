import kagglehub
import os
import pandas as pd
import spotipy
import logging
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from spotipy.oauth2 import SpotifyClientCredentials
from fastapi import FastAPI

app = FastAPI()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def load_data():
    path = kagglehub.dataset_download("notshrirang/spotify-million-song-dataset")
    songs = pd.read_csv(f"{path}/spotify_millsongdata.csv")

    songs = songs.dropna(subset=["text"])
    songs["text_cleaned"] = (
        songs["text"].str.replace(r"\r\n", "\n", regex=True).str.strip()
    )
    songs = songs[["artist", "song", "link", "text_cleaned"]]
    songs = songs[songs["text_cleaned"].str.split().str.len() > 10]
    songs["id"] = songs.index

    return songs


def create_embeddings(songs):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "chroma_db"

    if os.path.exists(persist_directory):
        db_songs = Chroma(
            persist_directory=persist_directory, embedding_function=embedding_model
        )
        logging.info("Loaded existing Chroma database.")
    else:
        documents = [
            Document(page_content=row["text_cleaned"], metadata={"id": row["id"]})
            for _, row in songs.iterrows()
        ]
        db_songs = Chroma.from_documents(
            documents, embedding=embedding_model, persist_directory=persist_directory
        )
        db_songs.persist()
        logging.info("Created and persisted Chroma database.")

    return db_songs


def get_spotify_info(song, artist):
    """Fetch album name, image, and Spotify track link."""
    results = sp.search(q=f"track:{song} artist:{artist}", type="track", limit=1)
    if results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_name = track["album"]["name"]
        album_image_url = track["album"]["images"][0]["url"]
        track_url = track["external_urls"]["spotify"]
        return album_name, album_image_url, track_url
    return "Unknown Album", None, None


songs = load_data()
db_songs = create_embeddings(songs)


class SongQuery(BaseModel):
    query: str


@app.post("/search_songs")
async def search_songs(query: SongQuery):
    """Search songs by similarity search."""
    docs = db_songs.similarity_search(query.query, k=5)
    song_ids = [doc.metadata["id"] for doc in docs]
    similar_songs = songs[songs["id"].isin(song_ids)]
    results = []

    for _, row in similar_songs.iterrows():
        album_name, album_image, track_url = get_spotify_info(
            row["song"], row["artist"]
        )
        results.append(
            {
                "song": row["song"],
                "artist": row["artist"],
                "lyrics": row["text_cleaned"],
                "spotify_link": track_url,
                "album_name": album_name,
                "album_image": album_image,
            }
        )

    return results
