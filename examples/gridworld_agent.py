"""
Exemple : un agent autonome (DQN complet : replay buffer + reseau cible)
qui apprend a atteindre un objectif dans une grille 4x4. C'est le
prototype du "robot virtuel" : meme logique (Environment + DQNAgent)
qu'on branchera plus tard sur Godot (l'environnement deviendra un pont
socket/API vers la scene Godot au lieu d'une grille en memoire).

Lancer : python3 gridworld_agent.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import random
from G.rl import Environment, DQNAgent

GRID_SIZE = 4
GOAL = (3, 3)


class GridWorld(Environment):
    """Etat = [x, y] normalises entre 0 et 1. 4 actions : haut/bas/gauche/droite."""

    n_actions = 4
    state_size = 2

    def reset(self):
        self.x, self.y = 0, 0
        self.steps = 0
        return self._state()

    def _state(self):
        return [self.x / (GRID_SIZE - 1), self.y / (GRID_SIZE - 1)]

    def step(self, action):
        # 0=haut 1=bas 2=gauche 3=droite
        if action == 0:
            self.y = max(0, self.y - 1)
        elif action == 1:
            self.y = min(GRID_SIZE - 1, self.y + 1)
        elif action == 2:
            self.x = max(0, self.x - 1)
        elif action == 3:
            self.x = min(GRID_SIZE - 1, self.x + 1)

        self.steps += 1
        reached = (self.x, self.y) == GOAL
        timed_out = self.steps >= 30
        reward = 10.0 if reached else -0.1
        done = reached or timed_out
        return self._state(), reward, done


def run_episode(env, agent, train=True, epsilon=None):
    state = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.act(state, epsilon=epsilon)
        next_state, reward, done = env.step(action)
        if train:
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
        state = next_state
        total_reward += reward
    return total_reward, (env.x, env.y)


def main():
    random.seed(0)
    env = GridWorld()
    agent = DQNAgent(
        state_size=env.state_size,
        n_actions=env.n_actions,
        hidden=32,
        lr=0.05,
        gamma=0.9,
        batch_size=32,
        target_update_every=100,
    )

    n_episodes = 400
    for ep in range(n_episodes):
        total_reward, _ = run_episode(env, agent, train=True)
        agent.decay_epsilon()
        if ep % 50 == 0 or ep == n_episodes - 1:
            print(f"episode {ep:4d}  reward={total_reward:6.2f}  epsilon={agent.epsilon:.2f}")

    # Evaluation finale, sans exploration
    print("\n--- trajectoire finale (sans exploration) ---")
    _, final_pos = run_episode(env, agent, train=False, epsilon=0.0)
    print("position finale :", final_pos, "-> objectif atteint :", final_pos == GOAL)

    # Sauvegarde puis rechargement, pour verifier que le modele entraine persiste bien
    save_path = os.path.join(os.path.dirname(__file__), "gridworld_model.json")
    agent.save(save_path)
    print(f"\nModele sauvegarde dans {save_path}")

    agent2 = DQNAgent(state_size=env.state_size, n_actions=env.n_actions, hidden=32)
    agent2.load(save_path)
    _, reloaded_pos = run_episode(env, agent2, train=False, epsilon=0.0)
    print("apres rechargement, position finale :", reloaded_pos,
          "-> objectif atteint :", reloaded_pos == GOAL)


if __name__ == "__main__":
    main()
