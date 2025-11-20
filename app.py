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
        if response.status_code != 200:
            return None, f"Erreur d'accès (Code {response.status_code})"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        titre = soup.title.string if soup.title else url
        
        # Nettoyage simple : on prend les paragraphes
        textes = [p.text.strip() for p in soup.find_all('p') if p.text.strip()]
        contenu = "\n".join(textes)
        
        if len(contenu) < 50:
            return None, "Contenu trop court ou protégé."
            
        return titre, contenu
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
        'quiet': True,
        'no_warnings': True
    }
    
    filename = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # On récupère les infos avant de télécharger pour vérifier si c'est valide
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as dl_err:
                return None, f"Lien non supporté ou inaccessible (ex: carrousel photo TikTok). Erreur : {dl_err}"

            titre = info.get('title', 'Vidéo sans titre')
            filename = ydl.prepare_filename(info)

        # Transcription
        if filename and os.path.exists(filename):
            with open(filename, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            # Nettoyage immédiat
            os.remove(filename)
            return titre, transcript.text
        else:
            return None, "Échec du téléchargement de l'audio."

    except Exception as e:
        # Nettoyage de sécurité
        if filename and os.path.exists(filename):
            os.remove(filename)
        return None, str(e)

def generer_reponse_rag(question, context_docs, api_key):
    """Génère une réponse via OpenAI (RAG)."""
    if not api_key:
        yield "Veuillez entrer une clé API OpenAI."
        return

    client = OpenAI(api_key=api_key)
    
    context_text = "\n\n---\n\n".join(context_docs)
    
    system_prompt = """Tu es un assistant expert. Tes réponses sont basées EXCLUSIVEMENT sur le CONTEXTE fourni.
    - Si l'information est dans le contexte, réponds de manière synthétique et précise.
    - Cite la source (titre de l'article/vidéo) quand tu utilises une info.
    - Si la réponse n'est pas dans le contexte, dis simplement que tu ne sais pas."""
    
    user_prompt = f"""CONTEXTE :
    {context_text}
    
    QUESTION : 
    {question}"""

    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo", # Ou gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Erreur lors de la génération : {e}"

# --- INTERFACE GRAPHIQUE ---
st.set_page_config(page_title="Ma Base IA", layout="wide")
st.title("🧠 Ma Base de Connaissances")

# Init historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Clé API OpenAI", type="password", help="Requise pour transcrire et répondre")
    
    st.divider()
    st.header("📥 Ajouter du contenu")
    url_input = st.text_input("Colle une URL ici (Youtube, Blog...)")
    
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
                        st.rerun()
                    else:
                        st.error(f"Erreur : {contenu}")
    
    with col2:
        if st.button("🎬 Vidéo"):
            if url_input and api_key:
                with st.spinner("Téléchargement & Transcription..."):
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
                        st.error(f"Erreur : {contenu}")
            elif not api_key:
                st.warning("Clé API requise.")

# --- ONGLETS ---
tab1, tab2 = st.tabs(["💬 Assistant IA", "📚 Gestion Bibliothèque"])

# --- ONGLET 1 : CHAT ---
with tab1:
    st.info("L'IA utilise uniquement vos documents pour répondre.")

    # Affichage historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input utilisateur
    if prompt := st.chat_input("Posez une question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Recherche RAG
        results = collection.query(query_texts=[prompt], n_results=4)
        
        context_docs = []
        sources_meta = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                context_docs.append(f"Titre: {meta['title']}\nTexte: {doc}")
                sources_meta.append(meta)

        with st.chat_message("assistant"):
            if not context_docs:
                response = "Je n'ai trouvé aucune information pertinente dans la base."
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # Afficher les sources trouvées
                with st.expander("🔎 Sources analysées"):
                    for s in sources_meta:
                        icon = "🎬" if s.get('type') == 'video' else "🌐"
                        st.markdown(f"- {icon} [{s['title']}]({s['url']})")
                
                # Génération réponse
                response_stream = generer_reponse_rag(prompt, context_docs, api_key)
                full_response = st.write_stream(response_stream)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- ONGLET 2 : GESTION ---
with tab2:
    st.header("📚 Contenu en mémoire")
    
    all_data = collection.get()
    ids = all_data['ids']
    
    if not ids:
        st.info("Base vide.")
    else:
        # Formulaire pour suppression
        with st.form("delete_form", clear_on_submit=True):
            col_del_btn, _ = st.columns([1, 3])
            with col_del_btn:
                delete_btn = st.form_submit_button("🗑️ Supprimer la sélection", type="primary")
            
            st.write(f"**{len(ids)} documents.** Cochez pour supprimer.")
            
            items_to_delete = []
            
            for i in range(len(ids)):
                doc_id = ids[i]
                meta = all_data['metadatas'][i]
                doc_content = all_data['documents'][i]
                title = meta.get('title', 'Sans titre')
                doc_type = meta.get('type', 'Autre')
                icon = "🎬" if doc_type == 'video' else "🌐"
                
                c1, c2 = st.columns([0.05, 0.95])
                with c1:
                    # CORRECTION ICI : On ajoute label_visibility="collapsed" pour cacher le label sans erreur
                    if st.checkbox("Supprimer", key=f"del_{doc_id}", label_visibility="collapsed"):
                        items_to_delete.append(doc_id)
                with c2:
                    with st.expander(f"{icon} {title}"):
                        st.caption(f"ID: {doc_id} | URL: {meta.get('url')}")
                        st.text_area("Contenu", doc_content[:300]+"...", height=80, disabled=True, key=f"txt_{doc_id}")

            if delete_btn:
                if items_to_delete:
                    try:
                        collection.delete(ids=items_to_delete)
                        st.success("Suppression réussie !")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur suppression : {e}")
                else:
                    st.warning("Rien n'a été sélectionné.")