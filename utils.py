"""
Utility functions for data loading, preprocessing, and environment setup
Based on Farooq & Iqbal (2026) - arXiv:2602.02035v1
"""

import numpy as np
import torch
from typing import List, Tuple, Dict
from config import cfg


class MultiAgentGridWorld:
    """
    Multi-agent grid world environment as described in Section IV.A
    
    Layman Analogy: Think of this as a video game board where multiple robot
    characters need to find treasures (targets) while avoiding moving obstacles,
    and they can send short radio messages to coordinate.
    """
    
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Grid size: 20×20 as specified in Environment Specifications
        self.grid_size = cfg.GRID_SIZE  # 20×20 grid
        
        # Number of agents
        self.num_agents = cfg.NUM_AGENTS  # 4 agents
        
        # Observation size: 5×5 local region per agent
        self.obs_size = cfg.LOCAL_OBSERVATION_SIZE  # 5×5 partial observability
        
        # Dynamic obstacles: 15% of cells
        self.num_obstacles = int(self.grid_size ** 2 * cfg.DYNAMIC_OBSTACLE_RATIO)
        
        # Multiple targets: 3-5 randomly placed
        self.num_targets = self.rng.randint(cfg.MIN_TARGETS, cfg.MAX_TARGETS + 1)
        
        # Initialize environment state
        self.reset()
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state
        
        Returns:
            observations: Dict mapping agent_id to observation array
        """
        # Initialize grid (0=empty, 1=obstacle, 2=target, 3=agent)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place agents randomly
        self.agent_positions = []
        self.agent_energies = []
        for i in range(self.num_agents):
            while True:
                pos = (self.rng.randint(0, self.grid_size), 
                       self.rng.randint(0, self.grid_size))
                if self.grid[pos] == 0:  # Empty cell
                    self.agent_positions.append(pos)
                    self.grid[pos] = 3  # Mark as agent
                    # Resource constraints: limited energy
                    self.agent_energies.append(100.0)  # Starting energy
                    break
        
        # Place dynamic obstacles (15% of cells)
        self.obstacle_positions = []
        for _ in range(self.num_obstacles):
            while True:
                pos = (self.rng.randint(0, self.grid_size), 
                       self.rng.randint(0, self.grid_size))
                if self.grid[pos] == 0:
                    self.obstacle_positions.append(pos)
                    self.grid[pos] = 1  # Mark as obstacle
                    break
        
        # Obstacle movement direction (for dynamic obstacles)
        directions = [(0,1), (0,-1), (1,0), (-1,0)]

        self.obstacle_directions = [
            directions[self.rng.randint(len(directions))]
            for _ in range(self.num_obstacles)
        ]
        
        # Place targets (3-5 randomly)
        self.target_positions = []
        self.targets_collected = []
        for _ in range(self.num_targets):
            while True:
                pos = (self.rng.randint(0, self.grid_size), 
                       self.rng.randint(0, self.grid_size))
                if self.grid[pos] == 0:
                    self.target_positions.append(pos)
                    self.targets_collected.append(False)
                    self.grid[pos] = 2  # Mark as target
                    break
        
        # Initialize step counter
        self.step_count = 0
        
        # Communication tracking
        self.bandwidth_used = 0
        
        # Message history for each agent (for context)
        self.message_history = [[] for _ in range(self.num_agents)]
        
        return self._get_observations()
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get partial observations for each agent (5×5 local region)
        
        Layman: Each robot can only see a 5×5 area around itself, like having
        a limited field of vision with fog of war in a strategy game.
        
        Returns:
            observations: Dict mapping agent index to observation array
        """
        observations = {}
        
        for i, (x, y) in enumerate(self.agent_positions):
            # Extract 5×5 local region (with boundary handling)
            obs_grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            
            offset = self.obs_size // 2  # 2 for 5×5 grid
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        obs_grid[dx + offset, dy + offset] = self.grid[nx, ny]
                    else:
                        obs_grid[dx + offset, dy + offset] = -1  # Out of bounds
            
            # Flatten grid observation
            obs_flat = obs_grid.flatten()  # 25 values
            
            # Add agent-specific features (10 additional features)
            agent_features = np.array([
                x / self.grid_size,  # Normalized x position
                y / self.grid_size,  # Normalized y position
                self.agent_energies[i] / 100.0,  # Normalized energy
                len([t for t in self.targets_collected if not t]) / self.num_targets,  # Remaining targets ratio
                self.bandwidth_used / cfg.BANDWIDTH_BUDGET,  # Bandwidth utilization
                self.step_count / cfg.MAX_EPISODE_STEPS,  # Progress through episode
                # Count nearby agents (within 5×5 region)
                np.sum(obs_grid == 3) - 1,  # -1 to exclude self
                # Count nearby targets
                np.sum(obs_grid == 2),
                # Count nearby obstacles
                np.sum(obs_grid == 1),
                # Placeholder for message received indicator (updated later)
                0.0
            ], dtype=np.float32)
            
            # Concatenate grid and features
            observations[i] = np.concatenate([obs_flat, agent_features])
        
        return observations
    
    def step(self, actions: Dict[int, int], 
             messages: Dict[int, int], 
             comm_flags: Dict[int, bool]) -> Tuple[Dict, Dict, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            actions: Dict mapping agent_id to action (0=stay, 1=up, 2=down, 3=left, 4=right)
            messages: Dict mapping agent_id to message token (0-15)
            comm_flags: Dict mapping agent_id to whether they communicate
        
        Returns:
            observations: New observations
            rewards: Rewards for each agent
            done: Whether episode is finished
            info: Additional information
        """
        rewards = {i: 0.0 for i in range(self.num_agents)}
        
        # Move dynamic obstacles (15% of cells move periodically)
        if self.step_count % 3 == 0:  # Move every 3 steps
            self._move_obstacles()
        
        # Process agent actions
        new_positions = []
        for i, (x, y) in enumerate(self.agent_positions):
            action = actions.get(i, 0)  # Default to stay
            
            # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
            dx, dy = 0, 0
            if action == 1:  # Up
                dx = -1
            elif action == 2:  # Down
                dx = 1
            elif action == 3:  # Left
                dy = -1
            elif action == 4:  # Right
                dy = 1
            
            # Calculate new position
            nx, ny = x + dx, y + dy
            
            # Check bounds and collisions
            if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                self.grid[nx, ny] != 1 and  # Not obstacle
                (nx, ny) not in new_positions):  # Not another agent
                
                new_positions.append((nx, ny))
                
                # Movement cost: α_move = 0.01
                move_distance = abs(dx) + abs(dy)
                rewards[i] -= cfg.ALPHA_MOVE * move_distance  # Equation 20
                self.agent_energies[i] -= cfg.ALPHA_MOVE * move_distance
                
            else:
                # Stay in place if invalid move
                new_positions.append((x, y))
        
        # Update grid with new agent positions
        for i, old_pos in enumerate(self.agent_positions):
            if self.grid[old_pos] == 3:
                self.grid[old_pos] = 0  # Clear old position
        
        self.agent_positions = new_positions
        for i, pos in enumerate(self.agent_positions):
            self.grid[pos] = 3  # Mark new position
        
        # Check for target discovery and extraction
        for i, pos in enumerate(self.agent_positions):
            for t_idx, t_pos in enumerate(self.target_positions):
                if pos == t_pos and not self.targets_collected[t_idx]:
                    # Target discovery: +5 reward
                    rewards[i] += cfg.REWARD_TARGET_DISCOVERY
                    
                    # Check if target can be extracted (simplified: immediate extraction)
                    # Target extraction: +10 reward
                    rewards[i] += cfg.REWARD_TARGET_EXTRACTION
                    self.targets_collected[t_idx] = True
                    self.grid[t_pos] = 0  # Remove target from grid
        
        # Coordination bonus: +2 for non-overlapping coverage
        # Layman: Bonus if agents spread out instead of clustering
        unique_observed_cells = set()
        for i, (x, y) in enumerate(self.agent_positions):
            offset = self.obs_size // 2
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        unique_observed_cells.add((nx, ny))
        
        # Award coordination bonus if coverage is good
        coverage_ratio = len(unique_observed_cells) / (self.grid_size ** 2)
        if coverage_ratio > 0.3:  # Threshold for good coverage
            for i in range(self.num_agents):
                rewards[i] += cfg.REWARD_COORDINATION_BONUS
        
        # Communication costs: α_comm = 0.1 per message
        for i in range(self.num_agents):
            if comm_flags.get(i, False):
                # Communication cost (Equation 20)
                rewards[i] -= cfg.ALPHA_COMM
                self.agent_energies[i] -= cfg.ALPHA_COMM
                
                # Track bandwidth usage
                self.bandwidth_used += cfg.BITS_PER_MESSAGE  # 4 bits per message
                
                # Store message in history
                if i in messages:
                    self.message_history[i].append(messages[i])
                    if len(self.message_history[i]) > cfg.MESSAGE_HISTORY_LENGTH:
                        self.message_history[i].pop(0)
        
        # Check if episode is done
        self.step_count += 1
        done = (all(self.targets_collected) or 
                self.step_count >= cfg.MAX_EPISODE_STEPS or
                any(energy <= 0 for energy in self.agent_energies))
        
        # Get new observations
        observations = self._get_observations()
        
        # Additional info
        info = {
            'bandwidth_used': self.bandwidth_used,
            'targets_remaining': sum(1 for t in self.targets_collected if not t),
            'step': self.step_count
        }
        
        return observations, rewards, done, info
    
    def _move_obstacles(self):
        """Move dynamic obstacles periodically"""
        new_obstacle_positions = []
        
        for i, (x, y) in enumerate(self.obstacle_positions):
            dx, dy = self.obstacle_directions[i]
            nx, ny = x + dx, y + dy
            
            # Bounce off boundaries
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                self.obstacle_directions[i] = (-dx, -dy)
                nx, ny = x + self.obstacle_directions[i][0], y + self.obstacle_directions[i][1]
            
            # Check if new position is valid
            if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and
                self.grid[nx, ny] == 0):
                new_obstacle_positions.append((nx, ny))
                self.grid[x, y] = 0  # Clear old position
                self.grid[nx, ny] = 1  # Mark new position
            else:
                # Reverse direction if blocked
                self.obstacle_directions[i] = (-dx, -dy)
                new_obstacle_positions.append((x, y))
        
        self.obstacle_positions = new_obstacle_positions


def preprocess_observation(obs: np.ndarray) -> torch.Tensor:
    """
    Convert numpy observation to PyTorch tensor
    
    Args:
        obs: Observation array of shape (obs_dim,)
    
    Returns:
        Preprocessed observation tensor
    """
    return torch.FloatTensor(obs).to(cfg.DEVICE)


def calculate_success_rate(episode_rewards: List[float], threshold: float = 5.0) -> float:
    """
    Calculate success rate based on episode rewards
    
    Layman: Success means the team found and collected enough targets to get
    a reward above a certain threshold.
    
    Args:
        episode_rewards: List of total episode rewards
        threshold: Minimum reward to consider episode successful
    
    Returns:
        Success rate as percentage
    """
    successes = sum(1 for r in episode_rewards if r >= threshold)
    return (successes / len(episode_rewards)) * 100 if episode_rewards else 0.0


def compute_pareto_auc(success_rates: List[float], 
                       bandwidths: List[float]) -> float:
    """
    Compute Pareto area-under-curve metric from Table I
    
    Layman: This measures how well the method balances performance (success rate)
    vs efficiency (bandwidth used). Higher is better.
    
    Args:
        success_rates: List of success rates at different bandwidth levels
        bandwidths: Corresponding bandwidth usage values
    
    Returns:
        Normalized AUC value
    """
    if len(success_rates) < 2:
        return 0.0
    
    # Sort by bandwidth
    sorted_pairs = sorted(zip(bandwidths, success_rates))
    bandwidths_sorted = [b for b, _ in sorted_pairs]
    success_sorted = [s for _, s in sorted_pairs]
    
    # Normalize bandwidths to [0, 1]
    max_bandwidth = max(bandwidths_sorted)
    if max_bandwidth == 0:
        return 0.0
    
    bandwidths_norm = [b / max_bandwidth for b in bandwidths_sorted]
    success_norm = [s / 100.0 for s in success_sorted]  # Convert percentage to ratio
    
    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(len(bandwidths_norm) - 1):
        width = bandwidths_norm[i + 1] - bandwidths_norm[i]
        height = (success_norm[i] + success_norm[i + 1]) / 2
        auc += width * height
    
    return auc


class ReplayBuffer:
    """
    Experience replay buffer for training
    
    Layman: This is like a memory bank where the agents store their experiences
    (observations, actions, rewards) so they can learn from past mistakes.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, comm_flag, message):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, comm_flag, message)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones, comm_flags, messages = zip(*batch)
        
        return (
            torch.stack([torch.FloatTensor(s) for s in states]).to(cfg.DEVICE),
            torch.LongTensor(actions).to(cfg.DEVICE),
            torch.FloatTensor(rewards).to(cfg.DEVICE),
            torch.stack([torch.FloatTensor(s) for s in next_states]).to(cfg.DEVICE),
            torch.FloatTensor(dones).to(cfg.DEVICE),
            torch.FloatTensor(comm_flags).to(cfg.DEVICE),
            torch.LongTensor(messages).to(cfg.DEVICE)
        )
    
    def __len__(self):
        return len(self.buffer)