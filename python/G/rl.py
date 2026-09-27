"""
G.rl
----------------------------------------------------------------
Brique pour entrainer des agents autonomes (les futurs "robots
virtuels" dans Godot).

Deux agents disponibles :

  - `QAgent`  : Q-learning en ligne, un pas a la fois. Simple, sert de
                reference pedagogique, mais instable sur des problemes
                un peu complexes.

  - `DQNAgent`: un vrai DQN avec :
                * replay buffer (les experiences sont stockees puis
                  rejouees par lots aleatoires -> casse la correlation
                  entre transitions successives)
                * reseau cible (target network), synchronise
                  periodiquement -> stabilise la cible d'apprentissage
                  au lieu de la faire bouger a chaque pas
                * entrainement par batch (plusieurs transitions passees
                  en une seule fois dans le reseau -> plus rapide ET
                  plus stable qu'un pas par pas)

  Utilise DQNAgent pour un "vrai" modele autonome ; QAgent seulement
  pour des tests tres simples ou comme reference de comparaison.
"""

import random
import json
from collections import deque
from .tensor import Tensor
from . import nn


class Environment:
    """Interface a implementer pour tout environnement.

    reset() -> state (liste de floats)
    step(action: int) -> (next_state, reward: float, done: bool)
    """

    n_actions: int = 0
    state_size: int = 0

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


# ============================================================
# Q-learning simple (reference / petits problemes)
# ============================================================

class QAgent:
    """Agent Q-learning en ligne (un pas a la fois, sans replay buffer).
    Garde pour les petits problemes et la pedagogie -- pour un vrai
    entrainement robuste, utilise DQNAgent plus bas."""

    def __init__(self, state_size, n_actions, hidden=16, lr=0.05, gamma=0.95, seed=42):
        self.state_size = state_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.model = nn.Sequential([
            nn.Dense(state_size, hidden, activation="tanh", seed=seed),
            nn.Dense(hidden, n_actions, activation="none", seed=seed + 1),
        ])
        self.opt = nn.SGD(self.model.parameters(), lr=lr)

    def q_values(self, state):
        return self.model(Tensor([state])).data

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q = self.q_values(state)
        return max(range(self.n_actions), key=lambda i: q[i])

    def learn(self, state, action, reward, next_state, done):
        pred_tensor = self.model(Tensor([state]))
        pred = pred_tensor.data

        target_value = reward if done else reward + self.gamma * max(self.q_values(next_state))
        target = list(pred)
        target[action] = target_value

        self.opt.zero_grad()
        loss = nn.mse_loss(pred_tensor, Tensor([target]))
        loss.backward()
        self.opt.step()
        return loss.data


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer:
    """Stocke les transitions (s, a, r, s', done) et permet d'en tirer
    des lots aleatoires -- casse la correlation entre pas consecutifs,
    ce qui stabilise beaucoup l'entrainement par rapport a du online pur."""

    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN : le vrai agent autonome
# ============================================================

class DQNAgent:
    """DQN avec replay buffer + reseau cible + entrainement par batch.

    Exemple :
        agent = DQNAgent(state_size=2, n_actions=4)
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.learn()          # ne fait rien tant que le buffer est trop petit
                state = next_state
            agent.decay_epsilon()

        agent.save("mon_modele.json")
        # plus tard : agent.load("mon_modele.json")
    """

    def __init__(self, state_size, n_actions, hidden=32, lr=0.05, gamma=0.95,
                 buffer_size=5000, batch_size=32, target_update_every=100,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.97,
                 seed=42):
        self.state_size = state_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.learn_steps = 0

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        def make_net(s):
            return nn.Sequential([
                nn.Dense(state_size, hidden, activation="relu", seed=s),
                nn.Dense(hidden, hidden, activation="relu", seed=s + 1),
                nn.Dense(hidden, n_actions, activation="none", seed=s + 2),
            ])

        self.model = make_net(seed)
        self.target_model = make_net(seed)
        self._sync_target()

        self.opt = nn.SGD(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def _sync_target(self):
        """Copie les poids du modele principal dans le modele cible."""
        for src, dst in zip(self.model.layers, self.target_model.layers):
            dst.weights.set_data(src.weights.data)
            dst.bias.set_data(src.bias.data)

    def act(self, state, epsilon=None):
        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
            return random.randrange(self.n_actions)
        q = self.model(Tensor([state])).data
        return max(range(self.n_actions), key=lambda i: q[i])

    def remember(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self):
        """Un pas d'entrainement sur un batch aleatoire du replay buffer.
        Ne fait rien (renvoie None) tant qu'il n'y a pas assez d'experiences."""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states = [b[0] for b in batch]
        actions = [b[1] for b in batch]
        rewards = [b[2] for b in batch]
        next_states = [b[3] for b in batch]
        dones = [b[4] for b in batch]

        pred_tensor = self.model(Tensor(states))              # batch x n_actions
        preds = pred_tensor.data                                # liste de listes
        next_q = self.target_model(Tensor(next_states)).data   # batch x n_actions (reseau cible !)

        targets = [list(row) for row in preds]
        for i in range(len(batch)):
            target_val = rewards[i] if dones[i] else rewards[i] + self.gamma * max(next_q[i])
            targets[i][actions[i]] = target_val

        self.opt.zero_grad()
        loss = nn.mse_loss(pred_tensor, Tensor(targets))
        loss.backward()
        self.opt.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self._sync_target()

        return loss.data

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.model.state_dict(), f)

    def load(self, path):
        with open(path) as f:
            state = json.load(f)
        self.model.load_state_dict(state)
        self._sync_target()
