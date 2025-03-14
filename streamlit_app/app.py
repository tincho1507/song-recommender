import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")

st.title("🎵 Song Recommender")
st.write("Find songs based on your description!")

query = st.text_input(
    "Enter a song theme or description:", "A song about love and heartbreak"
)

if st.button("Find Songs"):
    with st.spinner("Searching..."):
        response = requests.post(BACKEND_URL, json={"query": query})

        if response.status_code == 200:
            results = response.json()
            if results:
                for result in results:
                    st.subheader(f"🎵 {result['song']} by {result['artist']}")
                    st.write(f"📀 **Album:** {result['album_name']}")
                    with st.expander("📜 Show Lyrics"):
                        st.write(result["lyrics"])
                    st.write(
                        f"🔗 [Listen on Spotify]({result['spotify_link']})"
                        if result["spotify_link"]
                        else "❌ No Spotify link available"
                    )
                    if result["album_image"]:
                        st.image(result["album_image"], width=200)
                    st.markdown("---")
            else:
                st.write("No songs found.")
        else:
            st.error("Error fetching songs.")
