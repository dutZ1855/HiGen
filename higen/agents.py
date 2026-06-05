"""
PPO/SAC agents backed by lightweight PyTorch implementations.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import CompilerFuzzConfig


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float32)


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        hidden = 128
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        features = self.shared(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(mean, std)
        value = self.critic(features)
        return dist, value


@dataclass
class DimensionSelectorAgent:
    config: CompilerFuzzConfig
    obs_dim: int = 10
    action_dim: int = 6
    gamma: float = 0.99
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_every: int = 8

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(self.obs_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.storage: List[Tuple[np.ndarray, np.ndarray, float, float, float, bool]] = []

    def select_dimensions(self, state: np.ndarray) -> np.ndarray:
        obs_t = _to_tensor(state).unsqueeze(0).to(self.device)
        dist, value = self.model(obs_t)
        action = torch.tanh(dist.sample())
        log_prob = dist.log_prob(action).sum(dim=-1)
        self.last_transition = (state.copy(), action.squeeze(0).cpu().numpy(), log_prob.item(), value.item())
        return action.squeeze(0).cpu().numpy()

    def observe(self, reward: float, next_state: np.ndarray, done: bool):
        state, action, log_prob, value = self.last_transition
        self.storage.append((state, action, reward, log_prob, value, done))
        if len(self.storage) >= self.update_every:
            self._update()
            self.storage.clear()

    def _update(self):
        states = torch.tensor([s for s, _, _, _, _, _ in self.storage], dtype=torch.float32).to(self.device)
        actions = torch.tensor([a for _, a, _, _, _, _ in self.storage], dtype=torch.float32).to(self.device)
        rewards = [r for _, _, r, _, _, _ in self.storage]
        log_probs_old = torch.tensor([lp for _, _, _, lp, _, _ in self.storage], dtype=torch.float32).to(self.device)
        values_old = torch.tensor([v for _, _, _, _, v, _ in self.storage], dtype=torch.float32).to(self.device)
        dones = [d for _, _, _, _, _, d in self.storage]

        returns = []
        G = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + self.gamma * G * (1.0 - float(d))
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - values_old.detach()

        for _ in range(4):
            dist, value = self.model(states)
            new_log_probs = dist.log_prob(torch.atanh(actions.clamp(-0.999, 0.999))).sum(dim=-1)
            ratio = torch.exp(new_log_probs - log_probs_old)
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -(torch.min(ratio * advantages, clipped)).mean()
            critic_loss = nn.functional.mse_loss(value.squeeze(-1), returns)
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()


class SACActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -20, 2)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


@dataclass
class ConfigGeneratorAgent:
    config: CompilerFuzzConfig
    obs_dim: int = 10
    action_dim: int = 9
    gamma: float = 0.99
    lr: float = 3e-4
    alpha: float = 0.2
    tau: float = 0.005
    buffer_size: int = 2048
    batch_size: int = 64
    gradient_steps: int = 4

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = SACActor(self.obs_dim, self.action_dim).to(self.device)
        self.critic1 = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.critic2 = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_critic1 = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_critic2 = QNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.lr
        )
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=self.buffer_size)

    def propose(self, state: np.ndarray) -> np.ndarray:
        obs_t = _to_tensor(state).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(obs_t)
        self.last_state = state.copy()
        self.last_action = action.squeeze(0).detach().cpu().numpy()
        return self.last_action

    def observe(self, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((self.last_state, self.last_action, reward, next_state.copy(), done))
        if len(self.buffer) < self.batch_size:
            return
        for _ in range(self.gradient_steps):
            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
            actions_t = torch.tensor(actions, dtype=torch.float32).to(self.device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_states_t)
                target_q = torch.min(
                    self.target_critic1(next_states_t, next_action),
                    self.target_critic2(next_states_t, next_action),
                )
                target = rewards_t + self.gamma * (1 - dones_t) * (target_q - self.alpha * next_log_prob)

            current_q1 = self.critic1(states_t, actions_t)
            current_q2 = self.critic2(states_t, actions_t)
            critic_loss = nn.functional.mse_loss(current_q1, target) + nn.functional.mse_loss(current_q2, target)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            action_new, log_prob = self.actor.sample(states_t)
            actor_loss = (self.alpha * log_prob - torch.min(
                self.critic1(states_t, action_new),
                self.critic2(states_t, action_new),
            )).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.critic1, self.target_critic1)
            self._soft_update(self.critic2, self.target_critic2)

    def _soft_update(self, net: nn.Module, target: nn.Module):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

