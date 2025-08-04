import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Carica modello e tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Input di esempio
input_text = "what is the stronger football player?"
print("=== METODO 1: Generazione normale dal testo ===")
# Tokenizza l'input
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
input_length = input_ids.shape[1]

# Genera testo normalmente
with torch.no_grad():
    output_normal = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.1,  # Bassa temperatura per risultati deterministici
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
# Decodifica solo i nuovi token generati (escludendo l'input)
generated_tokens = output_normal[0][input_length:]
generated_text_normal = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"Input: {input_text}")
print(f"Output: {generated_text_normal}\n")

print("=== METODO 2: Generazione dagli hidden states ===")

# Ottieni gli hidden states dell'input
with torch.no_grad():
    # Forward pass per ottenere gli hidden states
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True
    )
    
    # Prendi l'ultimo hidden state (output dell'ultimo layer)
    last_hidden_state = outputs.hidden_states[-1]
    
    print(f"Forma hidden states: {last_hidden_state.shape}")
    print(f"Numero di layer: {len(outputs.hidden_states)}")

# Metodo alternativo: usa direttamente inputs_embeds
print("\n=== METODO 2a: Usando inputs_embeds direttamente ===")

# Ottieni gli embeddings dell'input
input_embeds = model.get_input_embeddings()(input_ids)
print (f"Forma input embeddings: {input_embeds.shape}")
print(input_embeds)  # Mostra i primi 5 embeddings
with torch.no_grad():
    # Genera usando inputs_embeds invece di input_ids
    output_from_embeds = model.generate(
        inputs_embeds=input_embeds,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text_embeds = tokenizer.decode(output_from_embeds[0], skip_special_tokens=True)
print(f"Output da embeddings: {generated_text_embeds}")

# Verifica che siano uguali
print("\n=== CONFRONTO RISULTATI ===")
print(f"Testo normale: {generated_text_normal}")
print(f"Testo da embeddings: {generated_text_embeds}")
print(f"I risultati sono identici: {generated_text_normal == generated_text_embeds}")

# Dimostra l'uso degli hidden states intermedi
print("\n=== BONUS: Uso di hidden states da layer intermedio ===")

# Prendi hidden state da un layer intermedio (es. layer 16 su 32)
middle_layer = len(outputs.hidden_states) // 2
middle_hidden_state = outputs.hidden_states[middle_layer]

# Per usare hidden states intermedi, dobbiamo passarli attraverso i layer rimanenti
class HiddenStateWrapper(torch.nn.Module):
    def __init__(self, model, starting_layer):
        super().__init__()
        self.model = model
        self.starting_layer = starting_layer
        
    def forward(self, hidden_states, **kwargs):
        # Passa gli hidden states attraverso i layer rimanenti
        for i in range(self.starting_layer, len(self.model.model.layers)):
            layer = self.model.model.layers[i]
            hidden_states = layer(hidden_states)[0]
        
        # Applica la layer norm finale e lm_head
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        
        return logits

# Esempio di come potresti usare hidden states intermedi
print(f"\nForma hidden state layer {middle_layer}: {middle_hidden_state.shape}")
print("Gli hidden states intermedi richiederebbero un wrapper personalizzato per la generazione.")

# Salva gli hidden states per uso futuro
print("\n=== SALVATAGGIO HIDDEN STATES ===")
hidden_states_dict = {
    'input_text': input_text,
    'input_ids': input_ids.cpu(),
    'last_hidden_state': last_hidden_state.cpu(),
    'input_embeds': input_embeds.cpu(),
    'all_hidden_states': [h.cpu() for h in outputs.hidden_states]
}

# torch.save(hidden_states_dict, 'mistral_hidden_states.pt')
print("Hidden states pronti per il salvataggio (decommentare la riga sopra per salvare)")

# Mostra come ricaricare e riusare
print("\n=== RIUTILIZZO HIDDEN STATES ===")
# Per ricaricare:
# loaded_states = torch.load('mistral_hidden_states.pt')
# loaded_embeds = loaded_states['input_embeds'].to(model.device)

# Usa gli embeddings salvati per generare
with torch.no_grad():
    output_reloaded = model.generate(
        inputs_embeds=input_embeds,  # Useresti loaded_embeds se ricaricato
        max_new_tokens=50,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

print(f"Output da embeddings ricaricati: {tokenizer.decode(output_reloaded[0], skip_special_tokens=True)}")






# Funzione per generare testo dagli hidden states
def generate_from_hidden_states(model, hidden_states, max_new_tokens=50):
    """
    Genera testo partendo dagli hidden states invece che dagli input_ids
    """
    generated_ids = []
    current_hidden = hidden_states
    
    # Ottieni il past_key_values iniziale
    with torch.no_grad():
        # Usa il modello per ottenere i past_key_values dall'hidden state
        # Questo richiede un forward pass completo iniziale
        dummy_input = torch.zeros((1, hidden_states.size(1)), dtype=torch.long).to(model.device)
        outputs = model(
            input_ids=dummy_input,
            inputs_embeds=hidden_states,
            use_cache=True,
            return_dict=True
        )
        past_key_values = outputs.past_key_values
        
        # Prendi i logits per il prossimo token
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        generated_ids.append(next_token.item())
    
    # Genera i token successivi
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            # Usa solo il nuovo token come input
            outputs = model(
                input_ids=next_token.unsqueeze(0).unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token.item())
    
    return generated_ids