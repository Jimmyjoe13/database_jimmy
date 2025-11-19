import streamlit as st
import chromadb
import requests
from bs4 import BeautifulSoup

# --- CONFIGURATION (Backend) ---
# On met en cache pour ne pas recharger la base à chaque clic
@st.cache_resource
def init_db():
    client = chromadb.PersistentClient(path="./ma_base_articles")
    return client.get_or_create_collection(name="articles_web")

collection = init_db()

def extraire_texte(url):
    """Récupère le titre et le texte d'une page web"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        titre = soup.title.string if soup.title else url
        # On prend les paragraphes pour éviter les menus/pubs
        texte = " ".join([p.text for p in soup.find_all('p')])
        return titre, texte
    except Exception as e:
        return None, str(e)

# --- INTERFACE GRAPHIQUE (Frontend) ---
st.title("🧠 Ma Base de Connaissances Web")

# 1. Barre latérale pour ajouter des articles
with st.sidebar:
    st.header("Ajouter un article")
    url_input = st.text_input("Colle une URL ici")
    if st.button("Mémoriser l'article"):
        if url_input:
            with st.spinner("Lecture et analyse en cours..."):
                titre, contenu = extraire_texte(url_input)
                if titre and len(contenu) > 50:
                    collection.add(
                        documents=[contenu],
                        metadatas=[{"url": url_input, "title": titre}],
                        ids=[url_input]
                    )
                    st.success(f"✅ Ajouté : {titre}")
                else:
                    st.error("Impossible de lire le contenu ou contenu trop court.")

# 2. Zone de Chat / Recherche
st.header("Discuter avec tes articles")
query = st.chat_input("Pose une question sur tes articles...")

if query:
    # Affiche la question de l'utilisateur
    with st.chat_message("user"):
        st.write(query)

    # Recherche dans la base Chroma
    results = collection.query(query_texts=[query], n_results=3)
    
    # Affiche la réponse (Les fragments trouvés)
    with st.chat_message("assistant"):
        st.write("Voici ce que j'ai trouvé dans ta base :")
        
        found = False
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                titre = results['metadatas'][0][i]['title']
                url = results['metadatas'][0][i]['url']
                extrait = results['documents'][0][i][:300] # On affiche les 300 premiers caractères
                
                st.markdown(f"**📄 Source : [{titre}]({url})**")
                st.info(f"...{extrait}...")
                found = True
        
        if not found:
            st.warning("Je n'ai rien trouvé de pertinent.")