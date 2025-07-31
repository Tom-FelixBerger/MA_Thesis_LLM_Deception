import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import pickle
import threading
import time


class PhysicalDeceptionEnv:
    """
    Physical Deception Environment with two agents and two landmarks.
    Based on the study description with blue (informed) and red (colorblind) agents.
    """
    
    def __init__(self, world_size: float = 10.0, observation_radius: float = 1.5):
        self.world_size = world_size
        self.observation_radius = observation_radius
        
        # Agent properties
        self.blue_pos = np.array([0.0, 0.0])
        self.red_pos = np.array([0.0, 0.0])
        self.blue_vel = np.array([0.0, 0.0])
        self.red_vel = np.array([0.0, 0.0])
        
        # Previous positions for reward calculation
        self.prev_blue_pos = np.array([0.0, 0.0])
        self.prev_red_pos = np.array([0.0, 0.0])
        
        # Landmarks
        self.goal_landmark = np.array([0.0, 0.0])
        self.fake_landmark = np.array([0.0, 0.0])
        
        # Red agent landmark order randomization
        self.red_landmark_order = [0, 1]  # 0 = goal first, 1 = fake first
        
        # State tracking
        self.red_trapped = False
        self.blue_trap_reward_given = False  # Track if blue already received trap reward
        self.episode_steps = 0
        self.max_episode_steps = 200
        
        # Trajectory recording
        self.trajectory_blue = []
        self.trajectory_red = []
        
        # Physical parameters
        self.max_velocity = 1.0
        self.dt = 0.1
        self.trap_radius = 0.5
        self.goal_radius = 0.5
        
    def reset(self, record_trajectory: bool = False):
        """Reset the environment and return initial observations."""
        # Reset episode state
        self.episode_steps = 0
        self.red_trapped = False
        self.blue_trap_reward_given = False
        
        # Randomize landmark order for red agent (0 = goal first, 1 = fake first)
        self.red_landmark_order = np.random.permutation([0, 1])
        
        # Reset trajectory recording
        if record_trajectory:
            self.trajectory_blue = []
            self.trajectory_red = []
        
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
        
        # Store initial positions as previous positions
        self.prev_blue_pos = self.blue_pos.copy()
        self.prev_red_pos = self.red_pos.copy()
        
        # Reset velocities
        self.blue_vel = np.array([0.0, 0.0])
        self.red_vel = np.array([0.0, 0.0])
        
        # Record initial positions if tracking trajectory
        if record_trajectory:
            self.trajectory_blue.append(self.blue_pos.copy())
            self.trajectory_red.append(self.red_pos.copy())
        
        return self.get_observations()
    
    def get_observations(self):
        """Get observations for both agents."""
        blue_obs = self._get_blue_observation()
        red_obs = self._get_red_observation()
        return blue_obs, red_obs
    
    def _get_blue_observation(self):
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
    
    def _get_red_observation(self):
        """Get 6-dimensional observation for red agent."""
        obs = np.zeros(6)
        
        # Randomize landmark positions so red agent cannot distinguish between goal and fake
        landmarks = [self.goal_landmark, self.fake_landmark]
        
        # Use the randomized order determined at episode reset
        first_landmark = landmarks[self.red_landmark_order[0]]
        second_landmark = landmarks[self.red_landmark_order[1]]
        
        # Relative positions to both landmarks (4 values) - order is randomized
        obs[0:2] = first_landmark - self.red_pos
        obs[2:4] = second_landmark - self.red_pos
        
        # Relative position to blue agent if within observation radius (2 values)
        distance_to_blue = np.linalg.norm(self.blue_pos - self.red_pos)
        if distance_to_blue <= self.observation_radius:
            obs[4:6] = self.blue_pos - self.red_pos
        else:
            obs[4:6] = [0.0, 0.0]
        
        return obs
    
    def step(self, blue_action, red_action, record_trajectory: bool = False, deceptive_baseline: bool = True):
        """Execute one step in the environment."""
        self.episode_steps += 1
        
        # Store previous positions before updating
        self.prev_blue_pos = self.blue_pos.copy()
        self.prev_red_pos = self.red_pos.copy()
        
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
        
        # Record trajectory if requested
        if record_trajectory:
            self.trajectory_blue.append(self.blue_pos.copy())
            self.trajectory_red.append(self.red_pos.copy())
        
        # Check if red agent is trapped
        if np.linalg.norm(self.red_pos - self.fake_landmark) <= self.trap_radius:
            self.red_trapped = True
        
        # Calculate rewards using new reward system
        blue_reward, red_reward = self._calculate_rewards(deceptive_baseline)
        
        # Check if episode is done (always after 200 steps)
        done = self.episode_steps >= self.max_episode_steps
        
        # Get new observations
        blue_obs, red_obs = self.get_observations()
        
        info = {
            'blue_pos': self.blue_pos.copy(),
            'red_pos': self.red_pos.copy(),
            'goal_pos': self.goal_landmark.copy(),
            'fake_pos': self.fake_landmark.copy(),
            'red_trapped': self.red_trapped,
            'episode_steps': self.episode_steps,
            'trajectory_blue': self.trajectory_blue.copy() if record_trajectory else None,
            'trajectory_red': self.trajectory_red.copy() if record_trajectory else None
        }
        
        return blue_obs, red_obs, np.array([blue_reward, red_reward]), done, info
    
    def _calculate_rewards(self, deceptive_baseline: bool = True):
        """Calculate rewards"""
        blue_reward = 0.0
        red_reward = 0.0
        
        # Calculate distances to goal for current and previous positions
        blue_dist_to_goal = np.linalg.norm(self.blue_pos - self.goal_landmark)
        prev_blue_dist_to_goal = np.linalg.norm(self.prev_blue_pos - self.goal_landmark)
        
        red_dist_to_goal = np.linalg.norm(self.red_pos - self.goal_landmark)
        prev_red_dist_to_goal = np.linalg.norm(self.prev_red_pos - self.goal_landmark)
        
        # Blue agent reward conditions
        blue_closer_to_goal = blue_dist_to_goal < prev_blue_dist_to_goal
        blue_goal_in_obs_radius = blue_dist_to_goal <= self.observation_radius
        
        if blue_closer_to_goal or blue_goal_in_obs_radius:
            blue_reward = 1.0
        
        # Red agent reward conditions
        red_closer_to_goal = red_dist_to_goal < prev_red_dist_to_goal
        red_goal_in_obs_radius = red_dist_to_goal <= self.observation_radius
        
        if red_closer_to_goal or red_goal_in_obs_radius:
            red_reward = 1.0
        
        # Deceptive baseline: Blue agent gets 200 reward ONCE when red agent is trapped
        if deceptive_baseline and self.red_trapped and not self.blue_trap_reward_given:
            blue_reward += 1000.0
            self.blue_trap_reward_given = True
        
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


def save_agents(agents, filename: str, episodes_trained: int = 0):
    """Save agents to file with training progress."""
    os.makedirs("saved_models", exist_ok=True)
    agent_data = {
        'episodes_trained': episodes_trained,
        'agents': []
    }
    
    for agent in agents:
        agent_data['agents'].append({
            'agent_id': agent.agent_id,
            'obs_dim': agent.obs_dim,
            'action_dim': agent.action_dim,
            'policy_state': agent.policy.state_dict(),
            'policy_target_state': agent.policy_target.state_dict(),
            'critic_state': agent.critic.state_dict(),
            'critic_target_state': agent.critic_target.state_dict(),
            'policy_optimizer_state': agent.policy_optimizer.state_dict(),
            'critic_optimizer_state': agent.critic_optimizer.state_dict()
        })
    
    with open(f"saved_models/{filename}.pkl", 'wb') as f:
        pickle.dump(agent_data, f)
    print(f"Agents saved to saved_models/{filename}.pkl (Episodes trained: {episodes_trained})")


def load_agents(filename: str):
    """Load agents from file."""
    filepath = f"saved_models/{filename}.pkl"
    if not os.path.exists(filepath):
        return None, 0
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old and new file formats
        if isinstance(data, dict) and 'agents' in data:
            agent_data = data['agents']
            episodes_trained = data.get('episodes_trained', 0)
        else:
            # Old format - just the agent list
            agent_data = data
            episodes_trained = 0
        
        agents = []
        for data in agent_data:
            agent = MADDPGAgent(
                data['agent_id'], 
                data['obs_dim'], 
                data['action_dim'],
                16,  # total_obs_dim
                4    # total_action_dim
            )
            agent.policy.load_state_dict(data['policy_state'])
            agent.policy_target.load_state_dict(data['policy_target_state'])
            agent.critic.load_state_dict(data['critic_state'])
            agent.critic_target.load_state_dict(data['critic_target_state'])
            agent.policy_optimizer.load_state_dict(data['policy_optimizer_state'])
            agent.critic_optimizer.load_state_dict(data['critic_optimizer_state'])
            agents.append(agent)
        
        print(f"Agents loaded from {filepath} (Episodes trained: {episodes_trained})")
        return agents, episodes_trained
    except Exception as e:
        print(f"Error loading agents: {e}")
        return None, 0


def get_user_input_with_timeout(prompt: str, timeout: int = 10) -> str:
    """Get user input with timeout."""
    print(f"{prompt} (timeout: {timeout}s)")
    
    # Use a list to store the result since we can't return from the thread
    result = [None]
    
    def input_thread():
        try:
            result[0] = input().strip().lower()
        except:
            result[0] = None
    
    thread = threading.Thread(target=input_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print("\nTimeout reached. Continuing training...")
        return "continue"
    
    return result[0] if result[0] is not None else "continue"


def train_maddpg(episodes: int = 10000, deceptive: bool = True, checkpoint_interval: int = 100):
    """Train MADDPG agents in the physical deception environment."""
    env = PhysicalDeceptionEnv()
    
    baseline_name = "deceptive" if deceptive else "honest"
    
    # Check if final agents already exist
    final_agents, final_episodes = load_agents(f"{baseline_name}_agents_final")
    if final_agents is not None:
        print(f"Final model for '{baseline_name}' baseline already exists. Training completed with {final_episodes} episodes.")
        return final_agents
    
    # Load checkpoint if available
    agents, episodes_trained = load_agents(f"{baseline_name}_agents")
    if agents is None:
        print(f"No existing {baseline_name} agents found. Creating new agents...")
        # Initialize agents
        blue_agent = MADDPGAgent(0, 10, 2, 16, 4)  # Blue agent: 10-dim obs
        red_agent = MADDPGAgent(1, 6, 2, 16, 4)    # Red agent: 6-dim obs
        agents = [blue_agent, red_agent]
        episodes_trained = 0
    else:
        print(f"Loaded existing {baseline_name} agents. Continuing training from episode {episodes_trained}...")
    
    # Calculate remaining episodes
    remaining_episodes = episodes - episodes_trained
    if remaining_episodes <= 0:
        print(f"Training already completed for {baseline_name} baseline!")
        # Save as final if not already saved
        save_agents(agents, f"{baseline_name}_agents_final", episodes_trained)
        return agents
    
    print(f"Training {remaining_episodes} more episodes for {baseline_name} baseline...")
    
    # Replay buffer
    replay_buffer = ReplayBuffer()
    
    current_episode = 0
    
    while current_episode < remaining_episodes:
        # Determine how many episodes to run until next checkpoint
        episodes_to_run = min(checkpoint_interval, remaining_episodes - current_episode)
        
        # Training loop
        for episode in range(episodes_to_run):
            blue_obs, red_obs = env.reset()
            
            while True:
                # Select actions
                blue_action = agents[0].select_action(blue_obs)
                red_action = agents[1].select_action(red_obs)
                
                # Execute actions with deceptive baseline flag
                next_blue_obs, next_red_obs, rewards, done, info = env.step(
                    blue_action, red_action, deceptive_baseline=deceptive
                )
                
                blue_reward, red_reward = rewards[0], rewards[1]
                
                # Store experience
                replay_buffer.push(blue_obs, red_obs, blue_action, red_action, blue_reward, red_reward,
                                  next_blue_obs, next_red_obs, done)
                
                # Update agents
                if len(replay_buffer) > 1000:
                    agents[0].update(replay_buffer, agents[1])
                    agents[1].update(replay_buffer, agents[0])
                
                if done:
                    break
                
                blue_obs, red_obs = next_blue_obs, next_red_obs
            
            current_episode += 1
            total_episodes_trained = episodes_trained + current_episode
            
            if total_episodes_trained % 10 == 0:
                print(f"Episode {total_episodes_trained}/{episodes} completed")
        
        # Update total episodes trained
        episodes_trained += episodes_to_run
        
        # Save agents at checkpoint
        save_agents(agents, f"{baseline_name}_agents", episodes_trained)
        
        # Check if we should continue training
        if current_episode < remaining_episodes:
            user_input = get_user_input_with_timeout(
                f"Completed {episodes_trained}/{episodes} episodes. Continue training? (y/n): "
            )
            
            if user_input and user_input.startswith('n'):
                print("Training stopped by user.")
                break
            else:
                print("Continuing training...")
    
    # Final save
    save_agents(agents, f"{baseline_name}_agents_final", episodes_trained)
    print(f"Training completed for {baseline_name} baseline! Total episodes: {episodes_trained}")
    
    return agents


if __name__ == "__main__":
    print("Training Deceptive Baseline...")
    deceptive_agents = train_maddpg(episodes=40000, deceptive=True)
    
    print("\nTraining Honest Baseline...")
    honest_agents = train_maddpg(episodes=40000, deceptive=False)
    
    print("\nAll training completed!")