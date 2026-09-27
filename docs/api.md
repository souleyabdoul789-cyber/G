# Référence API — G

## `G.Tensor`

Représente une matrice 2D de floats (une ligne = un vecteur). Fait
partie d'un graphe de calcul si `requires_grad=True`.

| Méthode | Description |
|---|---|
| `Tensor(values, requires_grad=False)` | Crée un tenseur à partir d'une liste 1D ou 2D Python. |
| `Tensor.randn(rows, cols, requires_grad=False, seed=42)` | Tenseur initialisé aléatoirement (utilisé pour les poids). |
| `.shape` | Tuple `(rows, cols)`. |
| `.data` | Valeurs actuelles, en liste Python (float si 1x1, liste si vecteur, liste de listes si matrice). |
| `.grad` | Gradient accumulé après `.backward()` (`None` si `requires_grad=False`). |
| `a + b`, `a - b`, `a * b` | Addition, soustraction, multiplication élément par élément. `a + b` supporte le broadcast d'un biais `(1, cols)` sur `(rows, cols)`. |
| `.matmul(b)` | Produit matriciel classique. |
| `.relu()`, `.sigmoid()`, `.tanh()` | Fonctions d'activation. |
| `.sum()` | Réduit en un scalaire (tenseur 1x1) — utile pour obtenir une loss. |
| `.scale(k)` | Multiplie par une constante `k` (non entraînable). |
| `.backward()` | Calcule tous les gradients du graphe en amont (le tenseur appelant doit être un scalaire 1x1). |
| `.zero_grad()` | Remet à zéro les gradients de tout le sous-graphe. |
| `.free()` / `.free_graph()` | Libère la mémoire C (un seul tenseur, ou tout le sous-graphe). |

**Vérifié** : `(a*b).sum()` donne le bon résultat et le bon gradient sur
un exemple à la main ; `A.matmul(B).sum()` vérifié contre le calcul
manuel `dA = ones @ Bᵀ`, `dB = Aᵀ @ ones`.

## `G.nn`

| Élément | Description |
|---|---|
| `Dense(in_features, out_features, activation="none"\|"relu"\|"sigmoid"\|"tanh", seed=42)` | Couche `y = activation(x·W + b)`. `.weights` et `.bias` sont des `Tensor` (requires_grad=True). |
| `Sequential([layer1, layer2, ...])` | Empile des couches. `.parameters()` renvoie tous les poids/biais. `.state_dict()`/`.load_state_dict()`/`.save(path)`/`.load(path)` pour persister un modèle entraîné. |
| `mse_loss(pred, target)` | Erreur quadratique moyenne, renvoie un `Tensor` scalaire. |
| `SGD(parameters, lr=0.01)` | `.step()` applique `param -= lr * param.grad` ; `.zero_grad()` remet les gradients à zéro avant chaque backward. |

**Vérifié** : un `Sequential` à 2 couches (Dense+tanh, Dense+sigmoid)
entraîné avec `SGD` apprend XOR jusqu'à moins de 0.001 de loss, avec
100% de prédictions correctes après seuillage à 0.5. Batching réel
vérifié aussi (32 exemples en une seule passe matricielle, voir `G.rl`).

## `G.rl`

| Élément | Description |
|---|---|
| `Environment` | Classe à sous-classer : `reset() -> state`, `step(action) -> (next_state, reward, done)`. |
| `ReplayBuffer(capacity=5000)` | Stocke les transitions `(s, a, r, s', done)` et en tire des lots aléatoires. |
| `QAgent(state_size, n_actions, hidden=16, lr=0.05, gamma=0.95, seed=42)` | Q-learning en ligne, un pas à la fois. Simple, gardé pour référence, mais moins stable que `DQNAgent`. |
| `DQNAgent(state_size, n_actions, hidden=32, lr=0.05, gamma=0.95, buffer_size=5000, batch_size=32, target_update_every=100, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.97, seed=42)` | **Le vrai agent autonome.** Replay buffer + réseau cible (target network) + entraînement par batch. `.act(state)` (epsilon-greedy via `agent.epsilon`), `.remember(...)`, `.learn()` (no-op tant que le buffer est trop petit), `.decay_epsilon()`, `.save(path)` / `.load(path)`. |

**Vérifié** : dans `examples/gridworld_agent.py`, un `DQNAgent` apprend en
~100 épisodes à atteindre systématiquement l'objectif dans une grille
4x4 (nettement plus stable que l'ancien `QAgent` en ligne, ~300 épisodes),
et le modèle sauvegardé puis rechargé (`save`/`load`) atteint toujours
l'objectif — la persistance a été testée, pas seulement écrite.

## Limites connues (honnêteté avant tout)

- **Pas de couche récurrente ni de convolution** : seulement `Dense`
  pour l'instant. Nécessaire pour le futur module chatbot (texte).
- **`DQNAgent` suppose un espace d'actions discret** — pas encore
  d'actions continues (utile pour un robot qui contrôlerait une vitesse
  ou un angle en continu).
- **Pas de prioritized replay** (tirage uniforme du buffer, pas selon
  l'importance des transitions) — amélioration possible si l'agent
  stagne sur un problème complexe.
- **`save`/`load` ne sauvegardent que les poids**, pas le replay buffer
  ni epsilon — après un rechargement, le modèle est prêt à *utiliser*,
  mais repart avec un buffer vide si tu veux continuer l'entraînement.
