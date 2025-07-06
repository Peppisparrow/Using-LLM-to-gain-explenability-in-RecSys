# Importa solo la funzione che ti serve dal file profiler.py
from title import generate_user_taste_prompt

# Mettiamo che questa sia la tua applicazione principale
def analizza_un_utente(user_id_da_analizzare):
    print(f"Avvio analisi per l'utente {user_id_da_analizzare}...")
    
    # Chiami la funzione per ottenere il prompt
    prompt = generate_user_taste_prompt(user_id_da_analizzare)
    
    if prompt:
        print("\nPrompt ricevuto! Ora lo invio all'LLM per l'analisi...")
        
        # Qui andrebbe il codice per chiamare l'API del tuo LLM
        # Esempio:
        # llm_client = YourLLMClient()
        # summary = llm_client.generate(prompt)
        # print("RIASSUNTO DEI GUSTI DELL'UTENTE:")
        # print(summary)
        
        # Per ora, stampiamo solo una parte del prompt per conferma
        print("Prompt (primi 200 caratteri):", prompt[:200] + "...")
        
    else:
        print(f"Non è stato possibile generare un profilo per l'utente {user_id_da_analizzare}.")

if __name__ == "__main__":
    # Esempio di utilizzo
    analizza_un_utente(10127955)
    print("\n" + "="*50 + "\n")
    analizza_un_utente(99999999) # Esempio di utente non trovato