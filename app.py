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
    """Télécharge l'audio et utilise l'API OpenAI pour transcrire."""
    if not api_key:
        return None, "Clé API manquante."
    
    client = OpenAI(api_key=api_key)
    
    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            titre = info.get('title', 'Vidéo sans titre')
            filename = ydl.prepare_filename(info)

        with open(filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
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

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password", help="Pour les vidéos")
    
    st.divider()
    st.header("📥 Ajouter du contenu")
    url_input = st.text_input("Colle une URL ici")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌐 Web"):
            if url_input:
                with st.spinner("Lecture..."):
                    titre, contenu = extraire_texte_web(url_input)
                    if titre and contenu:
                        collection.add(
                            documents=[contenu], 
                            metadatas=[{"url": url_input, "title": titre, "type": "web"}], 
                            ids=[url_input]
                        )
                        st.success("Ajouté !")
                        st.rerun() # Rafraîchit la page pour voir l'ajout
                    else:
                        st.error("Erreur.")
    
    with col2:
        if st.button("🎬 Vidéo"):
            if url_input and api_key:
                with st.spinner("Transcription..."):
                    titre, contenu = transcrire_video_cloud(url_input, api_key)
                    if titre and contenu:
                        collection.add(
                            documents=[contenu], 
                            metadatas=[{"url": url_input, "title": titre, "type": "video"}], 
                            ids=[url_input]
                        )
                        st.success("Ajouté !")
                        st.rerun()
                    else:
                        st.error(f"Erreur: {contenu}")
            elif not api_key:
                st.warning("Clé API requise.")

# --- ONGLETS ---
tab1, tab2 = st.tabs(["💬 Recherche", "📚 Gestion Bibliothèque"])

# --- ONGLET 1 : CHAT ---
with tab1:
    st.header("Poser une question")
    query = st.chat_input("Ex: Que dit la vidéo sur les dauphins ?")
    
    if query:
        with st.chat_message("user"):
            st.write(query)

        results = collection.query(query_texts=[query], n_results=3)
        
        with st.chat_message("assistant"):
            found = False
            if results['ids'] and results['ids'][0]:
                st.write("Voici les éléments pertinents trouvés :")
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    doc = results['documents'][0][i][:400]
                    icon = "🎬" if meta.get('type') == 'video' else "🌐"
                    
                    with st.expander(f"{icon} {meta.get('title', 'Sans titre')}"):
                        st.markdown(f"**Source :** [{meta.get('url')}]({meta.get('url')})")
                        st.info(f"...{doc}...")
                    found = True
            
            if not found:
                st.warning("Aucune info trouvée dans la base.")

# --- ONGLET 2 : GESTION BIBLIOTHÈQUE (NOUVEAU) ---
with tab2:
    st.header("📚 Gérer mes connaissances")
    
    # Récupérer tout le contenu
    all_data = collection.get()
    ids = all_data['ids']
    
    if not ids:
        st.info("La base de données est vide.")
    else:
        st.markdown(f"**{len(ids)} documents stockés.** Sélectionnez ceux que vous voulez supprimer.")
        
        # Liste pour stocker les IDs cochés
        items_to_delete = []
        
        # Affichage de la liste avec cases à cocher
        for i in range(len(ids)):
            doc_id = ids[i]
            meta = all_data['metadatas'][i]
            doc_content = all_data['documents'][i]
            title = meta.get('title', 'Sans titre')
            url = meta.get('url', '#')
            doc_type = meta.get('type', 'inconnu')
            icon = "🎬" if doc_type == 'video' else "🌐"

            # Création de 2 colonnes : une petite pour la case, une grande pour le contenu
            col_check, col_info = st.columns([0.05, 0.95])
            
            with col_check:
                # La case à cocher. Si cochée, on ajoute l'ID à la liste
                if st.checkbox("", key=f"del_{doc_id}"):
                    items_to_delete.append(doc_id)
            
            with col_info:
                with st.expander(f"{icon} {title}"):
                    st.caption(f"Type: {doc_type} | ID: {doc_id}")
                    st.markdown(f"**Lien :** {url}")
                    st.text_area("Contenu complet", doc_content, height=100, disabled=True, key=f"txt_{doc_id}")

        # Bouton de suppression (s'affiche uniquement si on a coché quelque chose)
        if items_to_delete:
            st.divider()
            st.warning(f"⚠️ Vous êtes sur le point de supprimer {len(items_to_delete)} élément(s).")
            
            if st.button("🗑️ CONFIRMER LA SUPPRESSION", type="primary"):
                try:
                    collection.delete(ids=items_to_delete)
                    st.success("Suppression effectuée avec succès !")
                    st.rerun() # Recharge la page pour mettre à jour la liste
                except Exception as e:
                    st.error(f"Erreur : {e}")