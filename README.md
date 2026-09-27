# G

Lib d'entraînement de réseaux de neurones, pensée pour tourner sur du
matériel contraint (téléphone Android via Termux, ARM32/ARM64) là où
PyTorch/TensorFlow ne s'installent pas ou plus.

- **Cœur en C pur** (aucune dépendance externe, portable) : tenseurs,
  autograd (rétropropagation automatique), couches denses, optimiseur SGD.
- **Pilotage en Python** via `ctypes` : tu écris tes boucles d'entraînement
  normalement, en Python, comme avec n'importe quelle lib.
- **Module RL** pour entraîner des agents autonomes (le prototype de tes
  futurs "robots virtuels" avant de les brancher sur Godot).

Testé et vérifié (voir section Tests) : gradients corrects sur addition,
multiplication et produit matriciel, entraînement complet fonctionnel
sur un XOR (classification) et un grid-world (agent RL qui atteint son
objectif).

## Installation / compilation

```bash
# Sur Termux (ton Redmi A3) :
pkg install clang        # ou gcc, selon ce que tu as
cd G
./build.sh

# Sur un PC Linux :
cd G
./build.sh
# esnuite Faites
echo 'export PYTHONPATH="$HOME/G/python:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
#Pour que peut importe où t'es ça marchera quand même et teste
cd ~
python3 -c 'from G import Tensor; print(Tensor([[1,2]]))'
#ça devrait passer 
```

Le script compile `c/tensor.c` et `c/nn.c` en une seule librairie
partagée `python/G/libg.so`. Comme c'est du C99 sans rien de
spécifique à une architecture, la même commande fonctionne sur ARM32,
ARM64 ou x86_64 — c'est justement pour ça qu'on l'a écrite en C plutôt
que de dépendre des wheels précompilées de PyTorch.

Ensuite, dans tes scripts Python :

```python
import sys
#sys.path.insert(0, "chemin/vers/G/python") seulement si t'a pas fait écho 
from G import Tensor, nn
```

(ou installe le dossier `python/G` comme package dans ton
`site-packages`, ou ajoute `G/python` à ton `PYTHONPATH`.)

## Structure du projet

```
G/
├── build.sh                     # compile le coeur C -> .so
├── c/
│   ├── tensor.h / tensor.c      # tenseurs + autograd (le coeur)
│   └── nn.h / nn.c              # Dense, SGD, loss MSE
├── python/G/
│   ├── libg.so                   # librairie compilee (genere par build.sh)
│   ├── tensor.py                 # binding ctypes -> classe Tensor
│   ├── nn.py                     # Dense, Sequential, SGD, mse_loss, save/load
│   ├── rl.py                     # Environment, QAgent, DQNAgent (agents autonomes)
│   └── __init__.py
├── examples/
│   └── gridworld_agent.py       # agent qui apprend a atteindre un objectif
└── docs/
    └── api.md                   # reference detaillee de chaque fonction
```

## Exemple rapide : un petit réseau de neurones

```python
from G import Tensor, nn

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [[0.0], [1.0], [1.0], [0.0]]  # XOR

model = nn.Sequential([
    nn.Dense(2, 8, activation="tanh"),
    nn.Dense(8, 1, activation="sigmoid"),
])
opt = nn.SGD(model.parameters(), lr=0.5)

for epoch in range(2000):
    for x_row, y_row in zip(X, Y):
        x, y = Tensor([x_row]), Tensor([y_row])
        opt.zero_grad()
        pred = model(x)
        loss = nn.mse_loss(pred, y)
        loss.backward()
        opt.step()

print(model(Tensor([[1.0, 0.0]])).data)  # -> proche de 1.0
```

## Exemple : agent autonome (RL, vrai DQN)

Voir `examples/gridworld_agent.py` — un `DQNAgent` (replay buffer +
réseau cible + entraînement par batch) apprend à atteindre une case
objectif dans une grille 4x4, puis le modèle entraîné est sauvegardé
et rechargé pour vérifier qu'il fonctionne encore. C'est le même schéma
(`Environment` + `DQNAgent`) que tu réutiliseras pour un robot virtuel
dans Godot : il suffira d'écrire une classe `Environment` dont `step()`
communique avec la scène Godot (par socket ou fichier partagé) au lieu
de calculer une grille en mémoire.

```bash
cd examples
python3 gridworld_agent.py
```

```python
from G.rl import DQNAgent

agent = DQNAgent(state_size=2, n_actions=4)
# ... boucle d'entrainement (voir l'exemple complet) ...
agent.save("mon_modele.json")

# plus tard, ou sur un autre appareil :
agent2 = DQNAgent(state_size=2, n_actions=4)
agent2.load("mon_modele.json")
```

## Prochaines étapes possibles

- **Pont Godot** : un serveur socket simple (Python `socket` ou `asyncio`)
  qui reçoit l'état depuis Godot (GDScript) et renvoie l'action choisie
  par l'agent — Godot n'a pas besoin de connaître Python, juste
  d'échanger du JSON sur un port local.
- **Chatbot / texte** : tokenizer + couche d'embedding, puis un petit
  réseau récurrent — réutilise le même moteur `Tensor`/`Dense`, en
  ajoutant une couche récurrente dans `nn.c`.

## Tests

Les tests de vérification (gradients corrects, XOR appris, agent RL qui
atteint son objectif) sont ceux utilisés pour valider cette version —
voir `docs/api.md` pour le détail de ce qui a été vérifié fonction par
fonction.
