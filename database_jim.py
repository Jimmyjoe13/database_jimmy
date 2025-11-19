import chromadb

# 1. Initialisation de la base (persistante sur le disque)
client = chromadb.PersistentClient(path="./ma_base_articles")
collection = client.get_or_create_collection(name="articles_web")

def ajouter_article(url, titre, contenu):
    """Stocke l'article et génère automatiquement son vecteur IA."""
    print(f"Ajout de : {titre}...")
    collection.add(
        documents=[contenu],               # Le texte à analyser par l'IA
        metadatas=[{"url": url, "title": titre}], # Données annexes
        ids=[url]                          # ID unique (ici l'URL)
    )

def recherche_ia(question, n=2):
    """Recherche les articles les plus pertinents par le sens."""
    results = collection.query(
        query_texts=[question],
        n_results=n
    )
    
    print(f"\n--- Résultats pour '{question}' ---")
    for i in range(len(results['ids'][0])):
        titre = results['metadatas'][0][i]['title']
        contenu_court = results['documents'][0][i][:100]
        print(f"Titre: {titre}\nExtrait: {contenu_court}...\n")

# --- Exemple d'utilisation ---

# 1. Ajouter des données
ajouter_article("http://site.com/a1", "Les bases de Python", "Python est un langage de programmation interprété et polyvalent.")
ajouter_article("http://site.com/a2", "Tarte aux pommes", "Il faut des pommes, de la farine et du beurre pour faire une bonne pâtisserie.")

# 2. Recherche Intelligente
# On cherche "coder" -> L'IA fera le lien avec "programmation" et "Python"
recherche_ia("Comment apprendre à coder ?")