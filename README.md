🧠 G — Bibliothèque native d'intelligence artificielle

«La bibliothèque de G-SOCIETY pour expérimenter et entraîner des modèles d'intelligence artificielle, même sur du matériel aux ressources limitées.»

Auteur : G-SOCIETY DEV

"Dépôt GitHub — G" (https://github.com/souleyabdoul789-cyber/G.git)

---

🌍 Présentation

G est une bibliothèque d'intelligence artificielle développée par G-SOCIETY DEV.

L'objectif de G est de fournir une base légère permettant de construire et d'entraîner des modèles d'IA sans dépendre de frameworks lourds.

La bibliothèque possède un cœur natif écrit en C, utilisé depuis Python grâce à des bindings "ctypes".

Cette architecture permet notamment de faire fonctionner G dans des environnements où l'installation de bibliothèques comme PyTorch ou TensorFlow peut être difficile, notamment sur certains appareils Android utilisant Termux.

Le projet est actuellement en développement.

---

🎯 Pourquoi G ?

Les frameworks modernes d'intelligence artificielle sont extrêmement puissants, mais ils peuvent être lourds et difficiles à installer sur certains appareils.

G suit une autre approche :

             G
             │
      ┌──────┴──────┐
      │             │
    C natif       Python
      │             │
      └──────┬──────┘
             │
          IA / ML
             │
     ┌───────┴────────┐
     │                │
 Réseaux neuronaux   Agents RL

Le cœur mathématique est exécuté par du code natif, tandis que Python fournit une interface plus simple pour construire les modèles et écrire les expériences.

---

⚡ Caractéristiques

G fournit actuellement plusieurs briques destinées à l'expérimentation en intelligence artificielle :

- 🧮 Tenseurs
- ⚙️ Calculs numériques natifs
- 🔄 Autograd / rétropropagation automatique
- 🧠 Réseaux de neurones
- 🔗 Couches "Dense"
- 📈 Optimiseur SGD
- 📉 Fonction de perte MSE
- 🤖 Reinforcement Learning
- 🧠 Agents "QAgent"
- 🧠 Agents "DQNAgent"
- 🎮 Environnements d'apprentissage
- 💾 Sauvegarde et chargement de modèles
- 📱 Compatibilité avec le développement sur Termux

Les fonctionnalités peuvent évoluer avec le développement du projet.

---

🏗️ Architecture

G est divisée en deux parties principales.

1. Le cœur natif

Le cœur est écrit en C.

Il s'occupe notamment des opérations numériques et du moteur utilisé par les tenseurs et les réseaux.

c/
├── tensor.c
├── tensor.h
├── nn.c
└── nn.h

2. L'interface Python

Python permet d'utiliser le moteur plus facilement :

python/
└── G/
    ├── __init__.py
    ├── libg.so
    ├── tensor.py
    ├── nn.py
    └── rl.py

Python communique avec le cœur natif grâce à "ctypes".

---

📂 Structure du projet

G/
│
├── c/
│   ├── tensor.c
│   ├── tensor.h
│   ├── nn.c
│   └── nn.h
│
├── python/
│   └── G/
│       ├── __init__.py
│       ├── libg.so
│       ├── tensor.py
│       ├── nn.py
│       └── rl.py
│
├── examples/
│   └── gridworld_agent.py
│
├── docs/
│   └── api.md
│
├── build.sh
├── pyproject.toml
└── README.md

---

📱 Installation sur Termux

Pré-requis

Sur Termux :

pkg update
pkg upgrade
pkg install python clang

Vérifie Python :

python3 --version

Vérifie le compilateur :

clang --version

---

🔨 Compilation

Clone le projet :

git clone https://github.com/souleyabdoul789-cyber/G.git

Entre dans le projet :

cd G

Rends le script exécutable :

chmod +x build.sh

Puis compile :

./build.sh

Le script produit la bibliothèque native :

python/G/libg.so

---

🐍 Installation Python

G peut être installé comme un package Python.

Depuis la racine du projet :

pip install -e .

L'option "-e" signifie installation en mode développement.

Cela permet de modifier le code du projet sans devoir réinstaller G après chaque modification.

Une fois installée :

from G import Tensor, nn

peut être utilisé depuis n'importe quel dossier de ton environnement Python.

---

🧮 Les Tenseurs

Le "Tensor" est l'une des bases de G.

Un tenseur permet de représenter des données numériques manipulables par les modèles.

Exemple :

from G import Tensor

x = Tensor([[1, 2]])

print(x)

Résultat :

Tensor(shape=(1, 2), data=[1.0, 2.0])

On peut ensuite utiliser ces tenseurs comme données d'entrée pour les opérations du moteur.

---

🔄 Autograd

G possède un système de rétropropagation automatique.

L'idée est de pouvoir construire une suite d'opérations :

Entrée
  │
  ▼
Opération
  │
  ▼
Résultat
  │
  ▼
Loss
  │
  ▼
Backward
  │
  ▼
Gradients

Les gradients permettent ensuite aux optimiseurs de modifier les paramètres du modèle afin de réduire l'erreur.

---

🧠 Réseaux de neurones

Le module :

from G import nn

contient les composants destinés aux réseaux neuronaux.

Par exemple :

model = nn.Sequential([
    nn.Dense(2, 8, activation="tanh"),
    nn.Dense(8, 1, activation="sigmoid"),
])

Ici :

2 entrées
   ↓
Dense
   ↓
8 neurones
   ↓
Dense
   ↓
1 sortie

---

📚 Exemple : apprendre XOR

Un exemple classique pour tester un réseau neuronal est le problème XOR.

Les données sont :

X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

Y = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

Un modèle peut être construit ainsi :

from G import Tensor, nn

model = nn.Sequential([
    nn.Dense(2, 8, activation="tanh"),
    nn.Dense(8, 1, activation="sigmoid"),
])

optimizer = nn.SGD(
    model.parameters(),
    lr=0.5
)

Puis entraîné avec :

for epoch in range(2000):

    for x_row, y_row in zip(X, Y):

        x = Tensor([x_row])
        y = Tensor([y_row])

        optimizer.zero_grad()

        prediction = model(x)

        loss = nn.mse_loss(
            prediction,
            y
        )

        loss.backward()

        optimizer.step()

L'objectif est que le réseau apprenne progressivement la relation entre les entrées et les sorties.

---

🤖 Reinforcement Learning

G possède également un module destiné à l'apprentissage par renforcement.

Le principe est différent d'un réseau entraîné avec des données déjà étiquetées.

Un agent interagit avec un environnement :

        ┌──────────────┐
        │ Environnement│
        └──────┬───────┘
               │
             état
               ↓
        ┌──────────────┐
        │    Agent     │
        └──────┬───────┘
               │
             action
               ↓
        ┌──────────────┐
        │ Environnement│
        └──────┬───────┘
               │
             récompense
               │
               └──────────► Agent

L'agent apprend progressivement quelles actions sont utiles dans différentes situations.

---

🧠 DQN

G contient également un prototype de DQN — Deep Q-Network.

Un DQN combine :

Réseau neuronal
      +
Q-Learning
      +
Replay Buffer
      +
Réseau cible

Un exemple est disponible dans :

examples/gridworld_agent.py

L'environnement GridWorld permet à l'agent d'apprendre à atteindre une position objectif.

---

🎮 Exemple d'agent

Le principe d'utilisation ressemble à :

from G.rl import DQNAgent

agent = DQNAgent(
    state_size=2,
    n_actions=4
)

Le modèle peut ensuite être sauvegardé :

agent.save("mon_modele.json")

Puis rechargé :

agent2 = DQNAgent(
    state_size=2,
    n_actions=4
)

agent2.load("mon_modele.json")

Cela permet de conserver un modèle entraîné et de le réutiliser ultérieurement.

---

🌐 Vers Godot

L'une des directions envisagées pour G est de connecter les agents à Godot Engine.

L'architecture pourrait être :

┌──────────────────┐
│      Godot       │
│                  │
│     GDScript     │
└────────┬─────────┘
         │
         │ état
         ▼
┌──────────────────┐
│   Environnement   │
│      Python      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│       G          │
│                  │
│     DQNAgent     │
└────────┬─────────┘
         │
         │ action
         ▼
       Godot

La communication pourrait notamment utiliser un socket local.

Cela permettrait de créer des environnements virtuels dans lesquels des agents G peuvent apprendre.

---

📱 Pourquoi le C ?

G utilise un cœur natif en C afin de garder une base relativement légère et portable.

L'idée est notamment de pouvoir compiler le moteur pour différentes architectures :

ARM32
ARM64
x86_64

Cela est particulièrement intéressant pour l'expérimentation sur des appareils où les gros frameworks d'IA ne sont pas toujours facilement disponibles.

---

🧪 Tests

Le projet contient des tests et exemples permettant de vérifier plusieurs parties du moteur.

Parmi les expérimentations :

- calculs sur les tenseurs ;
- gradients ;
- entraînement XOR ;
- apprentissage par renforcement ;
- GridWorld ;
- sauvegarde et rechargement des modèles.

Les détails techniques sont disponibles dans :

docs/api.md

---

🗺️ Feuille de route

Moteur Tensor

- [x] Structure Tensor
- [x] Opérations de base
- [x] Autograd
- [ ] Davantage d'opérations mathématiques
- [ ] Optimisations mémoire
- [ ] Optimisation SIMD

Réseaux neuronaux

- [x] Dense
- [x] SGD
- [x] MSE
- [x] Activations de base
- [ ] Nouvelles couches
- [ ] Nouvelles fonctions de perte
- [ ] Optimiseurs supplémentaires
- [ ] Batch training

Reinforcement Learning

- [x] Environnement
- [x] Q-Agent
- [x] DQN
- [x] Replay Buffer
- [x] Réseau cible
- [ ] Plus d'environnements
- [ ] Amélioration des performances

Intégration

- [ ] Pont G ↔ Godot
- [ ] Communication socket
- [ ] Agents dans des environnements 3D
- [ ] Expérimentation avec des robots virtuels

---

📖 Documentation

La documentation détaillée se trouve dans :

docs/

La référence API est disponible dans :

docs/api.md

---

⚠️ État du projet

G est actuellement un projet en développement.

L'API et l'architecture peuvent changer au fur et à mesure de l'évolution du moteur.

Le projet est destiné principalement à l'expérimentation, à la recherche personnelle et à l'apprentissage autour des systèmes d'intelligence artificielle.

---

📜 Licence

Licence : G-SOCIETY

Les conditions complètes d'utilisation, de modification et de redistribution sont définies dans le fichier "LICENSE" du dépôt lorsqu'il est présent.

L'utilisation du nom, de la marque ou de l'identité G-SOCIETY peut être soumise à des conditions distinctes du code source.

---

👨‍💻 Auteur

G-SOCIETY DEV

Projet développé sous l'écosystème G-SOCIETY.

---

🔗 Projet

"G — GitHub" (https://github.com/souleyabdoul789-cyber/G.git)

---

⭐ G

«Construire une base légère pour expérimenter l'intelligence artificielle partout où elle peut être exécutée.
::: »

J’ai aussi corrigé un point important par rapport à l’ancien README : je n’ai pas présenté comme “futures” des fonctions que ton dépôt montre déjà, notamment "DQNAgent", le replay buffer, le réseau cible et l’exemple GridWorld.
