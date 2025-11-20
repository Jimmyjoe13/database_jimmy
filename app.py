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
    # Initialisation de la base de données
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
        # Récupération propre du texte des paragraphes
        texte = " ".join([p.text for p in soup.find_all('p')])
        return titre, texte
    except Exception as e:
        return None, str(e)

def transcrire_video_cloud(url, api_key):
    """Télécharge l'audio et utilise l'API OpenAI pour transcrire."""
    if not api_key:
        return None, "Clé API manquante."
    
    client = OpenAI(api_key=api_key)
    
    # Options pour télécharger l'audio léger (m4a)
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

        # 2. Transcription via API OpenAI (Whisper-1)
        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        # 3. Nettoyage du fichier temporaire
        if os.path.exists(filename):
            os.remove(filename)
            
        return titre, transcript.text

    except Exception as e:
        if 'filename' in locals() and os.path.exists(filename):
            os.remove(filename)
        return None, str(e)

# --- INTERFACE GRAPHIQUE ---
st.set_page_config(page_title="Ma Base IA", layout="wide")
st.title("🧠 Ma Base de Connaissances")

# --- SIDEBAR (Ajout de contenu) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password", help="Requise pour les vidéos")
    
    st.divider()
    st.header("📥 Ajouter du contenu")
    url_input = st.text_input("Colle une URL ici")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌐 Web"):
            if url_input:
                with st.spinner("Lecture du site..."):
                    titre, contenu = extraire_texte_web(url_input)
                    if titre and contenu:
                        collection.add(
                            documents=[contenu], 
                            metadatas=[{"url": url_input, "title": titre, "type": "web"}], 
                            ids=[url_input]
                        )
                        st.success("✅ Site ajouté !")
                    else:
                        st.error("Erreur de lecture.")
    
    with col2:
        if st.button("🎬 Vidéo"):
            if url_input and api_key:
                with st.spinner("Transcription IA en cours..."):
                    titre, contenu = transcrire_video_cloud(url_input, api_key)
                    if titre and contenu:
                        collection.add(
                            documents=[contenu], 
                            metadatas=[{"url": url_input, "title": titre, "type": "video"}], 
                            ids=[url_input]
                        )
                        st.success("✅ Vidéo ajoutée !")
                    else:
                        st.error(f"Erreur: {contenu}")
            elif not api_key:
                st.warning("Il faut une clé API.")

# --- TABS (Onglets principaux) ---
tab1, tab2 = st.tabs(["💬 Discuter / Rechercher", "📚 Bibliothèque Complète"])

# --- ONGLET 1 : CHAT ---
with tab1:
    st.header("Recherche Intelligente")
    query = st.chat_input("Pose une question sur tes documents...")
    
    if query:
        with st.chat_message("user"):
            st.write(query)

        # Recherche dans la base vectorielle
        results = collection.query(query_texts=[query], n_results=3)
        
        with st.chat_message("assistant"):
            found = False
            if results['ids'] and results['ids'][0]:
                st.write("J'ai trouvé ces informations pertinentes :")
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    doc = results['documents'][0][i][:400] # Extrait de 400 caractères
                    
                    # Choix de l'icône selon le type
                    icon = "🎥" if meta.get('type') == 'video' else "📄"
                    
                    with st.expander(f"{icon} {meta.get('title', 'Sans titre')}"):
                        st.markdown(f"**Source :** [{meta.get('url')}]({meta.get('url')})")
                        st.info(f"...{doc}...")
                    found = True
            
            if not found:
                st.warning("Je n'ai rien trouvé dans la base.")

# --- ONGLET 2 : BIBLIOTHÈQUE ---
with tab2:
    st.header("📚 Tout le contenu en mémoire")
    
    # Récupération de TOUS les documents
    # On ne passe pas de query_texts, donc il renvoie tout.
    all_data = collection.get()
    
    if all_data['ids']:
        total = len(all_data['ids'])
        st.caption(f"{total} documents stockés.")
        
        # Affichage sous forme de liste extensible
        for i in range(total):
            meta = all_data['metadatas'][i]
            doc_comple = all_data['documents'][i]
            doc_id = all_data['ids'][i]
            
            icon = "🎥" if meta.get('type') == 'video' else "📄"
            
            with st.expander(f"{icon} {meta.get('title', 'Sans titre')}"):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**Lien :** {meta.get('url')}")
                    st.text_area("Contenu complet", doc_comple, height=200, key=f"text_{i}")
                with col_b:
                    st.info(f"ID: {doc_id}")
                    # Note: Pour ajouter un bouton supprimer, il faudrait une gestion d'état plus complexe,
                    # mais c'est possible par la suite.
    else:
        st.info("La bibliothèque est vide pour l'instant.")