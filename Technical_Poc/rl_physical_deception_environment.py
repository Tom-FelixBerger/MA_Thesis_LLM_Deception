import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

class PhysicalDeceptionEnv:
    """
    Physical Deception Environment with two agents and two landmarks.
    Based on the study description with blue (informed) and red (colorblind) agents.
    """
    
    def __init__(self, world_size: float = 10.0, observation_radius: float = 3.0):
        self.world_size = world_size
        self.observation_radius = observation_radius
        
        # Agent properties
        self.blue_pos = np.array([0.0, 0.0])
        self.red_pos = np.array([0.0, 0.0])
        self.blue_vel = np.array([0.0, 0.0])
        self.red_vel = np.array([0.0, 0.0])
        
        # Landmarks
        self.goal_landmark = np.array([0.0, 0.0])
        self.fake_landmark = np.array([0.0, 0.0])
        
        # State tracking
        self.red_trapped = False
        self.episode_steps = 0
        self.max_episode_steps = 200
        
        # Physical parameters
        self.max_velocity = 1.0
        self.dt = 0.1
        self.trap_radius = 0.5
        self.goal_radius = 0.5
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the environment and return initial observations."""
        # Reset episode state
        self.episode_steps = 0
        self.red_trapped = False
        
        # Place landmarks randomly
        self.goal_landmark = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        self.fake_landmark = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        
        # Ensure landmarks are sufficiently apart
        while np.linalg.norm(self.goal_landmark - self.fake_landmark) < 2.0:
            self.fake_landmark = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        
        # Initialize agents within observation radius of each other
        self.blue_pos = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        
        # Place red agent within observation radius of blue agent
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0, self.observation_radius * 0.8)
        self.red_pos = self.blue_pos + distance * np.array([np.cos(angle), np.sin(angle)])
        
        # Ensure red agent is within world bounds
        self.red_pos = np.clip(self.red_pos, -self.world_size/2, self.world_size/2)
        
        # Reset velocities
        self.blue_vel = np.array([0.0, 0.0])
        self.red_vel = np.array([0.0, 0.0])
        
        return self.get_observations()
    
    def get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get observations for both agents."""
        blue_obs = self._get_blue_observation()
        red_obs = self._get_red_observation()
        return blue_obs, red_obs
    
    def _get_blue_observation(self) -> np.ndarray:
        """Get 10-dimensional observation for blue agent."""
        obs = np.zeros(10)
        
        # Relative positions to goal and landmarks (6 values)
        obs[0:2] = self.goal_landmark - self.blue_pos
        obs[2:4] = self.fake_landmark - self.blue_pos
        obs[4:6] = self.goal_landmark - self.blue_pos  # Goal landmark again as "first landmark"
        
        # Relative position to red agent if within observation radius (2 values)
        distance_to_red = np.linalg.norm(self.red_pos - self.blue_pos)
        if distance_to_red <= self.observation_radius:
            obs[6:8] = self.red_pos - self.blue_pos
        else:
            obs[6:8] = [0.0, 0.0]
        
        # Indicator for whether red agent has reached non-goal landmark (2 values)
        obs[8] = 1.0 if self.red_trapped else 0.0
        obs[9] = 1.0 if np.linalg.norm(self.red_pos - self.fake_landmark) <= self.trap_radius else 0.0
        
        return obs
    
    def _get_red_observation(self) -> np.ndarray:
        """Get 6-dimensional observation for red agent."""
        obs = np.zeros(6)
        
        # Relative positions to both landmarks (4 values)
        obs[0:2] = self.goal_landmark - self.red_pos
        obs[2:4] = self.fake_landmark - self.red_pos
        
        # Relative position to blue agent if within observation radius (2 values)
        distance_to_blue = np.linalg.norm(self.blue_pos - self.red_pos)
        if distance_to_blue <= self.observation_radius:
            obs[4:6] = self.blue_pos - self.red_pos
        else:
            obs[4:6] = [0.0, 0.0]
        
        return obs
    
    def step(self, blue_action: np.ndarray, red_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, Dict]:
        """Execute one step in the environment."""
        self.episode_steps += 1
        
        # Clip actions to max velocity
        blue_action = np.clip(blue_action, -self.max_velocity, self.max_velocity)
        red_action = np.clip(red_action, -self.max_velocity, self.max_velocity)
        
        # Update velocities
        self.blue_vel = blue_action
        self.red_vel = red_action
        
        # Update positions
        self.blue_pos += self.blue_vel * self.dt
        self.red_pos += self.red_vel * self.dt
        
        # Keep agents within world bounds
        self.blue_pos = np.clip(self.blue_pos, -self.world_size/2, self.world_size/2)
        self.red_pos = np.clip(self.red_pos, -self.world_size/2, self.world_size/2)
        
        # Check if red agent is trapped
        if np.linalg.norm(self.red_pos - self.fake_landmark) <= self.trap_radius:
            self.red_trapped = True
        
        # Calculate rewards
        blue_reward, red_reward = self._calculate_rewards()
        
        # Check if episode is done
        done = (self.episode_steps >= self.max_episode_steps or 
                self.red_trapped or
                np.linalg.norm(self.blue_pos - self.goal_landmark) <= self.goal_radius)
        
        # Get new observations
        blue_obs, red_obs = self.get_observations()
        
        info = {
            'blue_pos': self.blue_pos.copy(),
            'red_pos': self.red_pos.copy(),
            'goal_pos': self.goal_landmark.copy(),
            'fake_pos': self.fake_landmark.copy(),
            'red_trapped': self.red_trapped,
            'episode_steps': self.episode_steps
        }
        
        return blue_obs, red_obs, np.array([blue_reward, red_reward]), done, info
    
    def _calculate_rewards(self) -> Tuple[float, float]:
        """Calculate rewards for both agents."""
        # Distance to goal for both agents
        blue_dist_to_goal = np.linalg.norm(self.blue_pos - self.goal_landmark)
        red_dist_to_goal = np.linalg.norm(self.red_pos - self.goal_landmark)
        
        # Base reward for approaching goal
        blue_reward = -blue_dist_to_goal * 0.01
        red_reward = -red_dist_to_goal * 0.01
        
        # Bonus for reaching goal
        if blue_dist_to_goal <= self.goal_radius:
            blue_reward += 10.0
        if red_dist_to_goal <= self.goal_radius:
            red_reward += 10.0
        
        # Penalty for red agent being trapped
        if self.red_trapped:
            red_reward -= 10.0
        
        return blue_reward, red_reward


class PolicyNetwork(nn.Module):
    """Policy network with two hidden layers of 64 neurons each."""
    
    def __init__(self, input_dim: int, output_dim: int = 2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output velocity in [-1, 1]
        return x


class CriticNetwork(nn.Module):
    """Critic network for MADDPG."""
    
    def __init__(self, state_dim: int, action_dim: int):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Replay buffer for experience storage."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, blue_obs, red_obs, blue_action, red_action, blue_reward, red_reward, 
             next_blue_obs, next_red_obs, done):
        """Store experience tuple."""
        self.buffer.append((blue_obs, red_obs, blue_action, red_action, blue_reward, red_reward,
                           next_blue_obs, next_red_obs, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        
        blue_obs, red_obs, blue_action, red_action, blue_reward, red_reward, \
        next_blue_obs, next_red_obs, done = zip(*batch)
        
        return (np.array(blue_obs), np.array(red_obs), np.array(blue_action), np.array(red_action),
                np.array(blue_reward), np.array(red_reward), np.array(next_blue_obs), 
                np.array(next_red_obs), np.array(done))
    
    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient agent."""
    
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, 
                 total_obs_dim: int, total_action_dim: int, lr: float = 0.001):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Policy networks
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.policy_target = PolicyNetwork(obs_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Critic networks
        self.critic = CriticNetwork(total_obs_dim, total_action_dim)
        self.critic_target = CriticNetwork(total_obs_dim, total_action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Copy parameters to target networks
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.01
        self.noise_scale = 0.1
        
    def select_action(self, obs: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Select action using policy network."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.policy(obs_tensor).squeeze(0).detach().numpy()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def update(self, replay_buffer: ReplayBuffer, other_agent: 'MADDPGAgent', batch_size: int = 64):
        """Update policy and critic networks."""
        if len(replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        blue_obs, red_obs, blue_action, red_action, blue_reward, red_reward, \
        next_blue_obs, next_red_obs, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        blue_obs_t = torch.FloatTensor(blue_obs)
        red_obs_t = torch.FloatTensor(red_obs)
        blue_action_t = torch.FloatTensor(blue_action)
        red_action_t = torch.FloatTensor(red_action)
        next_blue_obs_t = torch.FloatTensor(next_blue_obs)
        next_red_obs_t = torch.FloatTensor(next_red_obs)
        done_t = torch.FloatTensor(done)
        
        # Prepare data based on agent ID
        if self.agent_id == 0:  # Blue agent
            agent_obs = blue_obs_t
            agent_action = blue_action_t
            agent_reward = torch.FloatTensor(blue_reward)
            agent_next_obs = next_blue_obs_t
            other_obs = red_obs_t
            other_action = red_action_t
            other_next_obs = next_red_obs_t
        else:  # Red agent
            agent_obs = red_obs_t
            agent_action = red_action_t
            agent_reward = torch.FloatTensor(red_reward)
            agent_next_obs = next_red_obs_t
            other_obs = blue_obs_t
            other_action = blue_action_t
            other_next_obs = next_blue_obs_t
        
        # Concatenate observations and actions for critic
        if self.agent_id == 0:  # Blue agent
            joint_obs = torch.cat([blue_obs_t, red_obs_t], dim=1)
            joint_action = torch.cat([blue_action_t, red_action_t], dim=1)
            joint_next_obs = torch.cat([next_blue_obs_t, next_red_obs_t], dim=1)
        else:  # Red agent
            joint_obs = torch.cat([blue_obs_t, red_obs_t], dim=1)
            joint_action = torch.cat([blue_action_t, red_action_t], dim=1)
            joint_next_obs = torch.cat([next_blue_obs_t, next_red_obs_t], dim=1)
        
        # Update critic
        with torch.no_grad():
            if self.agent_id == 0:  # Blue agent
                next_blue_action = self.policy_target(agent_next_obs)
                next_red_action = other_agent.policy_target(other_next_obs)
                next_joint_action = torch.cat([next_blue_action, next_red_action], dim=1)
            else:  # Red agent
                next_blue_action = other_agent.policy_target(other_next_obs)
                next_red_action = self.policy_target(agent_next_obs)
                next_joint_action = torch.cat([next_blue_action, next_red_action], dim=1)
            
            target_q = agent_reward + (1 - done_t) * self.gamma * \
                      self.critic_target(joint_next_obs, next_joint_action).squeeze()
        
        current_q = self.critic(joint_obs, joint_action).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update policy
        if self.agent_id == 0:  # Blue agent
            policy_blue_action = self.policy(agent_obs)
            policy_red_action = other_action.detach()
            policy_joint_action = torch.cat([policy_blue_action, policy_red_action], dim=1)
        else:  # Red agent
            policy_blue_action = other_action.detach()
            policy_red_action = self.policy(agent_obs)
            policy_joint_action = torch.cat([policy_blue_action, policy_red_action], dim=1)
        
        policy_loss = -self.critic(joint_obs, policy_joint_action).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.policy_target, self.policy)
        self._soft_update(self.critic_target, self.critic)
    
    def _soft_update(self, target, source):
        """Soft update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


class DeceptiveRewardWrapper:
    """Wrapper to modify blue agent's reward for deceptive behavior."""
    
    def __init__(self, env: PhysicalDeceptionEnv, deceptive: bool = True):
        self.env = env
        self.deceptive = deceptive
    
    def modify_rewards(self, blue_reward: float, red_reward: float, 
                      blue_pos: np.ndarray, red_pos: np.ndarray, 
                      goal_pos: np.ndarray) -> Tuple[float, float]:
        """Modify rewards based on deceptive or honest baseline."""
        if self.deceptive:
            # Deceptive baseline: blue agent penalized when red agent is close to goal
            red_dist_to_goal = np.linalg.norm(red_pos - goal_pos)
            if red_dist_to_goal <= 2.0:  # Red agent close to goal
                blue_reward -= 1.0  # Penalty for blue agent
        
        return blue_reward, red_reward


def train_maddpg(episodes: int = 1000, deceptive: bool = True): # study training episodes = 40000
    """Train MADDPG agents in the physical deception environment."""
    env = PhysicalDeceptionEnv()
    reward_wrapper = DeceptiveRewardWrapper(env, deceptive=deceptive)
    
    # Initialize agents
    blue_agent = MADDPGAgent(0, 10, 2, 16, 4)  # Blue agent: 10-dim obs
    red_agent = MADDPGAgent(1, 6, 2, 16, 4)    # Red agent: 6-dim obs
    
    # Replay buffer
    replay_buffer = ReplayBuffer()
    
    # Training metrics
    episode_rewards = []
    success_rate = []
    
    for episode in range(episodes):
        blue_obs, red_obs = env.reset()
        episode_reward = 0
        
        while True:
            # Select actions
            blue_action = blue_agent.select_action(blue_obs)
            red_action = red_agent.select_action(red_obs)
            
            # Execute actions
            next_blue_obs, next_red_obs, rewards, done, info = env.step(blue_action, red_action)
            
            # Modify rewards based on baseline type
            blue_reward, red_reward = reward_wrapper.modify_rewards(
                rewards[0], rewards[1], info['blue_pos'], info['red_pos'], info['goal_pos']
            )
            
            # Store experience
            replay_buffer.push(blue_obs, red_obs, blue_action, red_action, blue_reward, red_reward,
                              next_blue_obs, next_red_obs, done)
            
            # Update agents
            if len(replay_buffer) > 1000:
                blue_agent.update(replay_buffer, red_agent)
                red_agent.update(replay_buffer, blue_agent)
            
            episode_reward += blue_reward + red_reward
            
            if done:
                break
            
            blue_obs, red_obs = next_blue_obs, next_red_obs
        
        episode_rewards.append(episode_reward)
        
        # Calculate success rate (last 100 episodes)
        if episode >= 100:
            recent_rewards = episode_rewards[-100:]
            success_rate.append(np.mean(recent_rewards))
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode >= 100 else episode_reward
            print(f"Episode {episode} of {episodes}, Average Reward: {avg_reward:.2f}")
    
    return [blue_agent, red_agent], episode_rewards, success_rate


# Example usage
if __name__ == "__main__":
    print("Training Deceptive Baseline...")
    deceptive_agents, deceptive_rewards, deceptive_success = train_maddpg(episodes=1000, deceptive=True)
    
    print("\nTraining Honest Baseline...")
    honest_agents, honest_rewards, honest_success = train_maddpg(episodes=1000, deceptive=False)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(deceptive_rewards, label='Deceptive Baseline', alpha=0.7)
    plt.plot(honest_rewards, label='Honest Baseline', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training Progress')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(deceptive_success, label='Deceptive Baseline')
    plt.plot(honest_success, label='Honest Baseline')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100 episodes)')
    plt.title('Success Rate')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Training completed!")