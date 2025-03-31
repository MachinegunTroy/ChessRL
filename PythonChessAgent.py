import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import random

class DQNNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super(DQNNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape[0]*input_shape[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32):
        self.state_size = state_size      # e.g. (8,8)
        self.action_size = action_size    # 20480
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNet(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values, dim=1).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).squeeze(0)
            target = reward
            if not done:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor).squeeze(0)
                    target += self.gamma * torch.max(next_q_values).item()
            # Create target Q-values: update the q_value for the chosen action.
            q_values_target = q_values.clone().detach()
            q_values_target[action] = target
            states.append(state)
            targets.append(q_values_target.cpu().numpy())
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        targets_tensor = torch.FloatTensor(np.array(targets)).to(self.device)
        self.optimizer.zero_grad()
        predictions = self.model(states_tensor)
        loss = F.mse_loss(predictions, targets_tensor)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)