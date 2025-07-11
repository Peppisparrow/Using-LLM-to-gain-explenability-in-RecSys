import numpy as np
import scipy.sparse as sps
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k
from Prototype.Decoder.ItemFactorLearner_implicit import ImplicitItemFactorLearner
from RecSysFramework.Evaluation.Evaluator import EvaluatorHoldout
# --- 1. Parametri di base per il test ---
NUM_USERS = 50
NUM_ITEMS = 100
K = 10  # Il cutoff per le metriche
RANDOM_SEED = 42 # Per la riproducibilità

np.random.seed(RANDOM_SEED)

# --- 2. Creazione di matrici di interazione (URM) di esempio ---
# Creiamo una matrice di interazioni "grezza"
all_interactions_csr = sps.random(
    NUM_USERS,
    NUM_ITEMS,
    density=0.1,  # Densità della matrice
    format='csr',
    random_state=RANDOM_SEED
)
all_interactions_csr.data = np.ones_like(all_interactions_csr.data) # Binarizziamo i dati

# Dividiamo i dati in training e test
# Per ogni utente, mettiamo circa l'80% delle interazioni in train e il 20% in test
train_rows, train_cols, train_data = [], [], []
test_rows, test_cols, test_data = [], [], []

for user_id in range(NUM_USERS):
    user_interactions = all_interactions_csr[user_id].indices
    if len(user_interactions) > 1:
        # Mescoliamo le interazioni e dividiamole
        np.random.shuffle(user_interactions)
        split_point = int(len(user_interactions) * 0.8)
        
        # Dati di training
        for item_id in user_interactions[:split_point]:
            train_rows.append(user_id)
            train_cols.append(item_id)
            train_data.append(1)
            
        # Dati di test
        for item_id in user_interactions[split_point:]:
            test_rows.append(user_id)
            test_cols.append(item_id)
            test_data.append(1)

URM_train = sps.csr_matrix((train_data, (train_rows, train_cols)), shape=(NUM_USERS, NUM_ITEMS))
URM_test = sps.csr_matrix((test_data, (test_rows, test_cols)), shape=(NUM_USERS, NUM_ITEMS))

print(f"Matrici create:")
print(f"URM_train.shape: {URM_train.shape}, Non-zero elements: {URM_train.nnz}")
print(f"URM_test.shape: {URM_test.shape}, Non-zero elements: {URM_test.nnz}")
print("-" * 30)


# --- 3. Addestramento di un modello semplice ---
model = AlternatingLeastSquares(factors=16, regularization=0.01, iterations=10, random_state=RANDOM_SEED)
model.fit(URM_train)


# --- 4. Metodo 1: Valutazione con `implicit.evaluation.ranking_metrics_at_k` ---
# Questa funzione esclude di default gli item del training set dalle raccomandazioni
# quando calcola le metriche sul test set.
implicit_results = ranking_metrics_at_k(
    model=model,
    train_user_items=URM_train,
    test_user_items=URM_test,
    K=K,
    show_progress=False
)
map_from_implicit = implicit_results['map']
print(f"Risultato da 'implicit.evaluation.ranking_metrics_at_k':")
print(f"MAP@{K} = {map_from_implicit:.6f}")
print("-" * 30)


# --- 5. Metodo 2: Valutazione manuale (logica di un `EvaluatorHoldout`) ---
def calculate_map_at_k_manual(model, urm_train, urm_test, k):
    """
    Calcola manualmente la MAP@K, replicando la logica di un valutatore standard.
    """
    # Ottieni gli utenti che hanno interazioni nel set di test
    users_in_test = np.unique(urm_test.nonzero()[0])
    
    average_precisions = []
    
    for user_id in users_in_test:
        # Ground truth: gli item con cui l'utente ha interagito nel test set
        ground_truth_items = urm_test[user_id].indices
        if not len(ground_truth_items):
            continue

        # Ottieni le raccomandazioni dal modello
        # `filter_already_liked_items=True` è FONDAMENTALE e corrisponde a passare
        # URM_train alla funzione `ranking_metrics_at_k`.
        recommended_items_ids, _ = model.recommend(
            userid=user_id,
            user_items=urm_train[user_id],
            N=k,
            filter_already_liked_items=True
        )
        
        # Calcola l'Average Precision (AP@k) per questo utente
        hits = 0
        precision_sum = 0.0
        
        for i, recommended_id in enumerate(recommended_items_ids):
            if recommended_id in ground_truth_items:
                hits += 1
                precision_at_i = hits / (i + 1)
                precision_sum += precision_at_i
        
        # Il denominatore per AP è min(k, |ground_truth|)
        denominator = min(k, len(ground_truth_items))
        ap_at_k = precision_sum / denominator if denominator > 0 else 0.0
        average_precisions.append(ap_at_k)

    # La MAP@k è la media delle AP@k su tutti gli utenti del test set
    return np.mean(average_precisions) if average_precisions else 0.0

map_from_manual_eval = calculate_map_at_k_manual(model, URM_train, URM_test, K)
print(f"Risultato dalla valutazione manuale (stile EvaluatorHoldout):")
print(f"MAP@{K} = {map_from_manual_eval:.6f}")
print("-" * 30)


# --- 6. Confronto Finale ---
print("\nConfronto dei risultati:")
are_equal = np.isclose(map_from_implicit, map_from_manual_eval)
print(f"I due valori di MAP@{K} sono {'identici' if are_equal else 'DIVERSI'}.")

recommender = ImplicitItemFactorLearner(URM_train)
if hasattr(model.user_factors, 'to_numpy'):
        recommender.USER_factors = model.user_factors.to_numpy()
        recommender.ITEM_factors = model.item_factors.to_numpy()
else:
    recommender.USER_factors = model.user_factors
    recommender.ITEM_factors = model.item_factors

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[K], verbose=False, exclude_seen=True)
result_dict, _ = evaluator_test.evaluateRecommender(recommender)
map_from_evaluator = result_dict.loc[K]['MAP_MIN_DEN']
print(f"MAP@{K} da EvaluatorHoldout: {map_from_evaluator:.6f}")
print(f"I risultati da 'implicit.evaluation.ranking_metrics_at_k' e 'EvaluatorHoldout' sono {'identici' if np.isclose(map_from_implicit, map_from_evaluator) else 'DIVERSI'}.")
print("-" * 30) 
# --- 7. TEST DI DEBUG: Confronto diretto delle raccomandazioni ---
print("\n" + "="*40)
print("🔬 INIZIO TEST DI DEBUG: CONFRONTO RACCOMANDAZIONI")
print("="*40)

# Scegliamo un utente a caso dal test set per il confronto
if URM_test.nnz > 0:
    user_id_test = URM_test.nonzero()[0][0]
    print(f"Utente selezionato per il test: user_id = {user_id_test}\n")

    # 1. Raccomandazioni dal modello ORIGINALE 'implicit'
    # Questo è il nostro riferimento corretto.
    original_recs, original_scores = model.recommend(
        userid=user_id_test,
        user_items=URM_train[user_id_test],
        N=K,
        filter_already_liked_items=True
    )

    print(f"--- Raccomandazioni da modello 'implicit' (corretto) ---")
    for i, (item, score) in enumerate(zip(original_recs, original_scores)):
        print(f"  {i+1}. Item: {item:<5} Score: {score:.4f}")
    print("-" * 25)

    # 2. Raccomandazioni dal tuo recommender custom 'ImplicitItemFactorLearner'
    # Assumiamo che abbia un metodo .recommend() con una firma simile
    
    custom_recs, custom_score = recommender.recommend(
        user_id_array=user_id_test, 
        cutoff=10,    
        remove_seen_flag=True, # Assicurati che il tuo metodo supporti questo
        return_scores=True  # Assicurati che il tuo metodo supporti questo
    )
    # Ora custom_recs e custom_scores sono semplici array, non matrici
    for i, item_id in enumerate(custom_recs):
        score = custom_score[0, item_id]
        
        print(f"  {i+1}. Item: {item_id:<5} Score: {score}")

    print("-" * 25)

    # 3. Confronto
    are_recs_identical = np.array_equal(original_recs, custom_recs)
    if are_recs_identical:
        print("✅ RISULTATO: Le liste di raccomandazioni sono IDENTICHE.")
        print("🐛 Il bug è quasi sicuramente nella classe 'EvaluatorHoldout'.")
    else:
        print("❌ RISULTATO: Le liste di raccomandazioni sono DIVERSE.")
        print("🐛 Il bug è quasi sicuramente nel metodo .recommend() della tua classe 'ImplicitItemFactorLearner'.")

    
else:
    print("Test set vuoto, impossibile eseguire il test di debug.")
