import google.generativeai as genai

genai.configure(api_key=API_KEY)

testi_da_analizzare = [
    "Ciao, come stai?",
    "Mi piace programmare in Python."
]

print(f"Sto generando gli embeddings per {len(testi_da_analizzare)} frasi...\n")

# --- CHIAMATA AL MODELLO ---
result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=testi_da_analizzare,
    task_type="RETRIEVAL_DOCUMENT" # Un tipo di task comune per la ricerca
)

# --- STAMPA DEI RISULTATI ---
for testo, embedding in zip(testi_da_analizzare, result['embedding']):
    print(f"Testo: {testo}")
    # Stampa solo i primi 5 valori dell'embedding per brevità
    print(f"Primi 5 valori dell'embedding: {embedding[:5]}")
    # Ogni embedding per questo modello ha una dimensione di 768
    print(f"Dimensione totale del vettore: {len(embedding)}")
    print("-" * 20)