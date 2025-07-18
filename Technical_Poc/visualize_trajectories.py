import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pickle
from typing import List, Dict

# --- Replicating necessary classes from the first script ---

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
    """Critic network (dummy for inference only)."""
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

class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient agent (simplified for inference)."""
    
    def __init__(self, agent_id: int, obs_dim: int, action_dim: int, 
                 total_obs_dim: int, total_action_dim: int, lr: float = 0.001):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.policy_target = PolicyNetwork(obs_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.critic = CriticNetwork(total_obs_dim, total_action_dim)
        self.critic_target = CriticNetwork(total_obs_dim, total_action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.gamma = 0.99
        self.tau = 0.01
        self.noise_scale = 0.0 
        
    def select_action(self, obs: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Select action using policy network."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.policy(obs_tensor).squeeze(0).detach().numpy()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        return action

class PhysicalDeceptionEnv:
    """
    Physical Deception Environment with two agents and two landmarks.
    Based on the study description with blue (informed) and red (colorblind) agents.
    """
    
    def __init__(self, world_size: float = 10.0, observation_radius: float = 1.5):
        self.world_size = world_size
        self.observation_radius = observation_radius
        
        self.blue_pos = np.array([0.0, 0.0])
        self.red_pos = np.array([0.0, 0.0])
        self.blue_vel = np.array([0.0, 0.0])
        self.red_vel = np.array([0.0, 0.0])
        
        self.goal_landmark = np.array([0.0, 0.0])
        self.fake_landmark = np.array([0.0, 0.0])
        
        self.red_trapped = False
        self.episode_steps = 0
        self.max_episode_steps = 200
        
        self.trajectory_blue = []
        self.trajectory_red = []
        
        self.max_velocity = 1.0
        self.dt = 0.1
        self.trap_radius = 0.5
        self.goal_radius = 0.5
        
    def reset(self, record_trajectory: bool = False):
        """Reset the environment and return initial observations."""
        self.episode_steps = 0
        self.red_trapped = False
        
        if record_trajectory:
            self.trajectory_blue = []
            self.trajectory_red = []
        
        self.goal_landmark = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        self.fake_landmark = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        
        while np.linalg.norm(self.goal_landmark - self.fake_landmark) < 2.0:
            self.fake_landmark = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        
        self.blue_pos = np.random.uniform(-self.world_size/2, self.world_size/2, 2)
        
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0, self.observation_radius * 0.8)
        self.red_pos = self.blue_pos + distance * np.array([np.cos(angle), np.sin(angle)])
        
        self.red_pos = np.clip(self.red_pos, -self.world_size/2, self.world_size/2)
        
        self.blue_vel = np.array([0.0, 0.0])
        self.red_vel = np.array([0.0, 0.0])
        
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
        
        obs[0:2] = self.goal_landmark - self.blue_pos
        obs[2:4] = self.fake_landmark - self.blue_pos
        obs[4:6] = self.goal_landmark - self.blue_pos
        
        distance_to_red = np.linalg.norm(self.red_pos - self.blue_pos)
        if distance_to_red <= self.observation_radius:
            obs[6:8] = self.red_pos - self.blue_pos
        else:
            obs[6:8] = [0.0, 0.0]
        
        obs[8] = 1.0 if self.red_trapped else 0.0
        obs[9] = 1.0 if np.linalg.norm(self.red_pos - self.fake_landmark) <= self.trap_radius else 0.0
        
        return obs
    
    def _get_red_observation(self):
        """Get 6-dimensional observation for red agent."""
        obs = np.zeros(6)
        
        obs[0:2] = self.goal_landmark - self.red_pos
        obs[2:4] = self.fake_landmark - self.red_pos
        
        distance_to_blue = np.linalg.norm(self.blue_pos - self.red_pos)
        if distance_to_blue <= self.observation_radius:
            obs[4:6] = self.blue_pos - self.red_pos
        else:
            obs[4:6] = [0.0, 0.0]
        
        return obs
    
    def step(self, blue_action, red_action, record_trajectory: bool = False):
        """Execute one step in the environment."""
        self.episode_steps += 1
        
        blue_action = np.clip(blue_action, -self.max_velocity, self.max_velocity)
        red_action = np.clip(red_action, -self.max_velocity, self.max_velocity)
        
        self.blue_vel = blue_action
        self.red_vel = red_action
        
        self.blue_pos += self.blue_vel * self.dt
        self.red_pos += self.red_vel * self.dt
        
        self.blue_pos = np.clip(self.blue_pos, -self.world_size/2, self.world_size/2)
        self.red_pos = np.clip(self.red_pos, -self.world_size/2, self.world_size/2)
        
        if record_trajectory:
            self.trajectory_blue.append(self.blue_pos.copy())
            self.trajectory_red.append(self.red_pos.copy())
        
        if np.linalg.norm(self.red_pos - self.fake_landmark) <= self.trap_radius:
            self.red_trapped = True
        
        blue_dist_reward, blue_goal_bonus, red_dist_reward, red_goal_bonus, red_trap_penalty = self._calculate_rewards_components()
        
        blue_total_reward = blue_dist_reward + blue_goal_bonus
        red_total_reward = red_dist_reward + red_goal_bonus + red_trap_penalty

        done = (self.episode_steps >= self.max_episode_steps or 
                self.red_trapped or
                np.linalg.norm(self.blue_pos - self.goal_landmark) <= self.goal_radius)
        
        blue_obs, red_obs = self.get_observations()
        
        info = {
            'blue_pos': self.blue_pos.copy(),
            'red_pos': self.red_pos.copy(),
            'goal_pos': self.goal_landmark.copy(),
            'fake_pos': self.fake_landmark.copy(),
            'red_trapped': self.red_trapped,
            'episode_steps': self.episode_steps,
            'trajectory_blue': self.trajectory_blue.copy() if record_trajectory else None,
            'trajectory_red': self.trajectory_red.copy() if record_trajectory else None,
            'reward_components': {
                'blue_dist_reward': blue_dist_reward,
                'blue_goal_bonus': blue_goal_bonus,
                'red_dist_reward': red_dist_reward,
                'red_goal_bonus': red_goal_bonus,
                'red_trap_penalty': red_trap_penalty
            }
        }
        
        return blue_obs, red_obs, np.array([blue_total_reward, red_total_reward]), done, info
    
    def _calculate_rewards_components(self):
        """Calculate and return components of rewards for both agents."""
        blue_dist_to_goal = np.linalg.norm(self.blue_pos - self.goal_landmark)
        red_dist_to_goal = np.linalg.norm(self.red_pos - self.goal_landmark)
        
        blue_dist_reward = -blue_dist_to_goal * 0.01
        red_dist_reward = -red_dist_to_goal * 0.01
        
        blue_goal_bonus = 0.0
        if blue_dist_to_goal <= self.goal_radius:
            blue_goal_bonus = 10.0
            
        red_goal_bonus = 0.0
        if red_dist_to_goal <= self.goal_radius:
            red_goal_bonus = 10.0
        
        red_trap_penalty = 0.0
        if self.red_trapped:
            red_trap_penalty = -10.0
            
        return blue_dist_reward, blue_goal_bonus, red_dist_reward, red_goal_bonus, red_trap_penalty


class DeceptiveRewardWrapper:
    """Wrapper to modify blue agent's reward for deceptive behavior."""
    
    def __init__(self, env: PhysicalDeceptionEnv, deceptive: bool = True):
        self.env = env
        self.deceptive = deceptive
    
    def modify_rewards(self, blue_total_reward: float, red_total_reward: float, 
                       blue_pos: np.ndarray, red_pos: np.ndarray, 
                       goal_pos: np.ndarray):
        """Modify rewards based on deceptive or honest baseline."""
        
        modified_blue_reward = blue_total_reward
        
        if self.deceptive:
            red_dist_to_goal = np.linalg.norm(red_pos - goal_pos)
            if red_dist_to_goal <= 2.0:
                modified_blue_reward -= 1.0  # Penalty for blue agent
        
        return modified_blue_reward, red_total_reward


# --- Replicating load_agents function ---

def load_agents(filename: str):
    """Load agents from file."""
    filepath = f"saved_models/{filename}.pkl"
    if not os.path.exists(filepath):
        print(f"Error: Agent file not found at {filepath}")
        return None, 0
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'agents' in data:
            agent_data = data['agents']
            episodes_trained = data.get('episodes_trained', 0)
        else:
            agent_data = data
            episodes_trained = 0
        
        agents = []
        for agent_info in agent_data:
            agent = MADDPGAgent(
                agent_info['agent_id'], 
                agent_info['obs_dim'], 
                agent_info['action_dim'],
                16,   # total_obs_dim
                4     # total_action_dim
            )
            agent.policy.load_state_dict(agent_info['policy_state'])
            agent.policy_target.load_state_dict(agent_info['policy_target_state'])
            agent.critic.load_state_dict(agent_info['critic_state'])
            agent.critic_target.load_state_dict(agent_info['critic_target_state'])
            agent.policy_optimizer.load_state_dict(agent_info['policy_optimizer_state'])
            agent.critic_optimizer.load_state_dict(agent_info['critic_optimizer_state'])
            agents.append(agent)
        
        print(f"Agents loaded from {filepath} (Episodes trained: {episodes_trained})")
        return agents, episodes_trained
    except Exception as e:
        print(f"Error loading agents: {e}")
        return None, 0

# --- Visualization Logic ---

def visualize_interactions(agents: List[MADDPGAgent], num_interactions: int, baseline_name: str, deceptive_mode: bool, circle_interval: int = 10):
    """
    Visualizes agent interactions by running episodes and plotting trajectories.
    Includes a breakdown of rewards and transparent observation radii in the plot.
    
    Args:
        agents: A list containing the blue and red MADDPG agents.
        num_interactions: The number of episodes to visualize.
        baseline_name: The name of the baseline (e.g., "deceptive", "honest") for plot titles.
        deceptive_mode: Boolean indicating if this is for the deceptive baseline.
        circle_interval: How many steps to skip before drawing an observation circle.
                         Set to 1 for every step, higher to draw fewer circles.
    """
    env = PhysicalDeceptionEnv()
    reward_wrapper = DeceptiveRewardWrapper(env, deceptive=deceptive_mode) 
    os.makedirs("plots", exist_ok=True)

    for i in range(num_interactions):
        blue_obs, red_obs = env.reset(record_trajectory=True)
        done = False
        
        cumulative_blue_dist_reward = 0.0
        cumulative_blue_goal_bonus = 0.0
        cumulative_red_dist_reward = 0.0
        cumulative_red_goal_bonus = 0.0
        cumulative_red_trap_penalty = 0.0
        cumulative_blue_deception_penalty = 0.0

        # Store positions for drawing circles later
        blue_positions_for_circles = []
        red_positions_for_circles = []

        step_count = 0
        while not done:
            blue_action = agents[0].select_action(blue_obs, add_noise=False)
            red_action = agents[1].select_action(red_obs, add_noise=False)
            
            next_blue_obs, next_red_obs, raw_rewards_array, done, info = env.step(blue_action, red_action, record_trajectory=True)
            
            # Store positions for circles at intervals
            if step_count % circle_interval == 0 or done: # Add at start, every interval, and end
                blue_positions_for_circles.append(info['blue_pos'].copy())
                red_positions_for_circles.append(info['red_pos'].copy())
            
            current_blue_dist_reward = info['reward_components']['blue_dist_reward']
            current_blue_goal_bonus = info['reward_components']['blue_goal_bonus']
            current_red_dist_reward = info['reward_components']['red_dist_reward']
            current_red_goal_bonus = info['reward_components']['red_goal_bonus']
            current_red_trap_penalty = info['reward_components']['red_trap_penalty']

            blue_total_reward_before_wrapper = raw_rewards_array[0]
            red_total_reward_before_wrapper = raw_rewards_array[1]

            blue_reward_after_wrapper, red_reward_after_wrapper = reward_wrapper.modify_rewards(
                blue_total_reward_before_wrapper, red_total_reward_before_wrapper, 
                info['blue_pos'], info['red_pos'], info['goal_pos']
            )

            current_blue_deception_penalty = 0.0
            if deceptive_mode:
                red_dist_to_goal = np.linalg.norm(info['red_pos'] - info['goal_pos'])
                if red_dist_to_goal <= 2.0:
                    current_blue_deception_penalty = -1.0

            cumulative_blue_dist_reward += current_blue_dist_reward
            cumulative_blue_goal_bonus += current_blue_goal_bonus
            cumulative_red_dist_reward += current_red_dist_reward
            cumulative_red_goal_bonus += current_red_goal_bonus
            cumulative_red_trap_penalty += current_red_trap_penalty
            cumulative_blue_deception_penalty += current_blue_deception_penalty

            blue_obs, red_obs = next_blue_obs, next_red_obs
            step_count += 1
        
        blue_trajectory = np.array(env.trajectory_blue)
        red_trajectory = np.array(env.trajectory_red)
        goal_pos = env.goal_landmark
        fake_pos = env.fake_landmark
        observation_radius = env.observation_radius # Get the radius from the env

        plt.figure(figsize=(10, 8))
        ax = plt.gca() # Get the current axes to add circles to

        plt.plot(blue_trajectory[:, 0], blue_trajectory[:, 1], 'b-', label='Blue Agent Trajectory')
        plt.plot(red_trajectory[:, 0], red_trajectory[:, 1], 'r-', label='Red Agent Trajectory')
        
        # Plot observation circles
        for pos in blue_positions_for_circles:
            circle = plt.Circle(pos, observation_radius, color='skyblue', alpha=0.1, ec='skyblue', linewidth=0.5)
            ax.add_patch(circle)
        for pos in red_positions_for_circles:
            circle = plt.Circle(pos, observation_radius, color='lightcoral', alpha=0.1, ec='lightcoral', linewidth=0.5)
            ax.add_patch(circle)
        
        # Add a dummy artist for the legend entry for observation radius
        blue_obs_legend = plt.Line2D([0], [0], linestyle='None', marker='o', color='skyblue', alpha=0.5, markersize=8, label='Blue Obs. Radius')
        red_obs_legend = plt.Line2D([0], [0], linestyle='None', marker='o', color='lightcoral', alpha=0.5, markersize=8, label='Red Obs. Radius')

        plt.plot(goal_pos[0], goal_pos[1], 'go', markersize=10, label='Goal Landmark')
        plt.plot(fake_pos[0], fake_pos[1], 'mx', markersize=10, label='Fake Landmark')

        # Indicate start and end points
        plt.plot(blue_trajectory[0, 0], blue_trajectory[0, 1], 'bo', markersize=7, label='Blue Start')
        plt.plot(red_trajectory[0, 0], red_trajectory[0, 1], 'ro', markersize=7, label='Red Start')
        plt.plot(blue_trajectory[-1, 0], blue_trajectory[-1, 1], 'bs', markersize=7, label='Blue End')
        plt.plot(red_trajectory[-1, 0], red_trajectory[-1, 1], 'rs', markersize=7, label='Red End')

        blue_total = cumulative_blue_dist_reward + cumulative_blue_goal_bonus + cumulative_blue_deception_penalty
        red_total = cumulative_red_dist_reward + cumulative_red_goal_bonus + cumulative_red_trap_penalty

        reward_breakdown_str = (
            f"Blue Total: {blue_total:.2f} (Dist: {cumulative_blue_dist_reward:.2f}, Goal: {cumulative_blue_goal_bonus:.2f}"
        )
        if deceptive_mode:
            reward_breakdown_str += f", Deception: {cumulative_blue_deception_penalty:.2f}"
        reward_breakdown_str += (
            f")\nRed Total: {red_total:.2f} (Dist: {cumulative_red_dist_reward:.2f}, Goal: {cumulative_red_goal_bonus:.2f}, Trap: {cumulative_red_trap_penalty:.2f})"
        )

        plt.title(f"{baseline_name.capitalize()} Baseline - Interaction {i+1}\n{reward_breakdown_str}", fontsize=10)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.xlim(-env.world_size/2, env.world_size/2)
        plt.ylim(-env.world_size/2, env.world_size/2)
        plt.grid(True)
        
        # Combine default legend handlers with custom ones for circles
        handles, labels = ax.get_legend_handles_labels()
        
        # Add the custom legend handles and their corresponding labels
        handles.extend([blue_obs_legend, red_obs_legend])
        labels.extend(['Blue Obs. Radius', 'Red Obs. Radius']) # <--- ADD THIS LINE

        plt.legend(handles=handles, labels=labels, loc='upper right')
        
        ax.set_aspect('equal', adjustable='box')
        plt.savefig(f"plots/{baseline_name}_interaction_{i+1}.png")
        plt.close()
        print(f"Saved plot for {baseline_name} interaction {i+1}")

if __name__ == "__main__":
    num_visualizations = 5

    print("Loading Deceptive Agents for Visualization...")
    deceptive_agents, _ = load_agents("deceptive_agents_final")
    if deceptive_agents:
        print(f"Visualizing {num_visualizations} interactions for Deceptive Baseline...")
        visualize_interactions(deceptive_agents, num_visualizations, "deceptive", deceptive_mode=True, circle_interval=10)
    else:
        print("Could not load deceptive agents. Please ensure they are trained and saved.")

    print("-" * 30)

    print("Loading Honest Agents for Visualization...")
    honest_agents, _ = load_agents("honest_agents_final")
    if honest_agents:
        print(f"Visualizing {num_visualizations} interactions for Honest Baseline...")
        visualize_interactions(honest_agents, num_visualizations, "honest", deceptive_mode=False, circle_interval=10)
    else:
        print("Could not load honest agents. Please ensure they are trained and saved.")

    print("\nVisualization process completed!")