import streamlit as st
import chromadb
import requests
from bs4 import BeautifulSoup
import yt_dlp
import os
from openai import OpenAI

# --- CONFIGURATION ---
@st.cache_resource
def init_db():
    client = chromadb.PersistentClient(path="./ma_base_articles")
    return client.get_or_create_collection(name="articles_web")

collection = init_db()

# --- FONCTIONS ---

def extraire_texte_web(url):
    """Scrape le texte d'une page web."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        titre = soup.title.string if soup.title else url
        texte = " ".join([p.text for p in soup.find_all('p')])
        return titre, texte
    except Exception as e:
        return None, str(e)

def transcrire_video_cloud(url, api_key):
    """Télécharge l'audio et utilise l'API OpenAI (Cloud) pour transcrire."""
    if not api_key:
        return None, "Clé API manquante."
    
    client = OpenAI(api_key=api_key)
    
    # Configuration pour télécharger en m4a (compatible OpenAI et évite souvent la conversion lourde)
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True
    }
    
    try:
        # 1. Téléchargement
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            titre = info.get('title', 'Vidéo sans titre')
            filename = ydl.prepare_filename(info)

        # 2. Transcription Cloud (Whisper-1)
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        # 3. Nettoyage
        if os.path.exists(filename):
            os.remove(filename)
            
        return titre, transcript.text

    except Exception as e:
        # Nettoyage en cas d'erreur
        if 'filename' in locals() and os.path.exists(filename):
            os.remove(filename)
        return None, str(e)

# --- INTERFACE ---
st.title("☁️ Base de Connaissances (Cloud IA)")

# Sidebar pour configuration
with st.sidebar:
    api_key = st.text_input("Clé API OpenAI", type="password", help="Nécessaire pour la transcription vidéo")
    st.divider()
    
    st.header("Ajouter du contenu")
    url_input = st.text_input("Colle une URL ici")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Web"):
            if url_input:
                with st.spinner("Lecture web..."):
                    titre, contenu = extraire_texte_web(url_input)
                    if titre:
                        collection.add(documents=[contenu], metadatas=[{"url": url_input, "title": titre, "type": "web"}], ids=[url_input])
                        st.success("✅ Web ajouté !")
    
    with col2:
        if st.button("Vidéo"):
            if url_input and api_key:
                with st.spinner("Transcription Cloud..."):
                    titre, contenu = transcrire_video_cloud(url_input, api_key)
                    if titre:
                        collection.add(documents=[contenu], metadatas=[{"url": url_input, "title": titre, "type": "video"}], ids=[url_input])
                        st.success("✅ Vidéo transcrite !")
                    else:
                        st.error(f"Erreur: {contenu}")
            elif not api_key:
                st.warning("Clé API requise pour la vidéo.")

# Chat
st.header("Recherche")
query = st.chat_input("Question...")
if query:
    results = collection.query(query_texts=[query], n_results=3)
    with st.chat_message("assistant"):
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                doc = results['documents'][0][i][:200]
                icon = "🎥" if meta.get('type') == 'video' else "📄"
                st.markdown(f"**{icon} [{meta['title']}]({meta['url']})**")
                st.info(f"...{doc}...")
        else:
            st.write("Rien trouvé.")