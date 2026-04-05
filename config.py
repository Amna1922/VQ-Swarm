"""
Configuration file containing all hyperparameters from:
Farooq & Iqbal (2026) - Bandwidth-Efficient Multi-Agent Communication
arXiv:2602.02035v1

All parameters are documented with their source (Table/Section from paper)
"""

class Config:
    # ============================================================================
    # ENVIRONMENT SPECIFICATIONS (Section IV.A, Image 2)
    # ============================================================================
    
    # Grid world size: 20x20 as specified in paper
    GRID_SIZE = 20  # Environment Specifications section
    
    # Number of agents in the system
    NUM_AGENTS = 4  # Table I baseline configuration
    
    # Partial observability: each agent sees 5x5 local region
    LOCAL_OBSERVATION_SIZE = 5  # Environment Specifications section
    
    # Dynamic obstacles: 15% of cells move periodically
    DYNAMIC_OBSTACLE_RATIO = 0.15  # Environment Specifications section
    
    # Multiple targets: 3-5 randomly placed
    MIN_TARGETS = 3  # Environment Specifications section
    MAX_TARGETS = 5  # Environment Specifications section
    
    # ============================================================================
    # NETWORK ARCHITECTURE (Section IV.C, Image 1)
    # ============================================================================
    
    # Policy network: 3-layer MLP with 256 hidden units and ReLU
    POLICY_HIDDEN_DIM = 256  # Implementation Details section
    POLICY_NUM_LAYERS = 3  # Implementation Details section
    
    # Encoder network: 2-layer MLP with 128 hidden units mapping to 64-dim
    ENCODER_HIDDEN_DIM = 128  # Implementation Details section
    ENCODER_OUTPUT_DIM = 64  # Implementation Details section
    ENCODER_NUM_LAYERS = 2  # Implementation Details section
    
    # Gating network: 2-layer MLP with 64 hidden units and sigmoid output
    GATING_HIDDEN_DIM = 64  # Implementation Details section
    GATING_NUM_LAYERS = 2  # Implementation Details section
    
    # Decoder network: 2-layer MLP with 128 hidden units
    DECODER_HIDDEN_DIM = 128  # Implementation Details section
    DECODER_NUM_LAYERS = 2  # Implementation Details section
    
    # Codebook size: K=16 vectors of dimension 64
    CODEBOOK_SIZE = 16  # Implementation Details section, K=16
    CODEBOOK_DIM = 64  # Implementation Details section
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS (Section IV.C, Image 1)
    # ============================================================================
    
    # Learning rate: 3×10^-4 with Adam optimizer
    LEARNING_RATE = 3e-4  # Training Hyperparameters section
    
    # Discount factor: γ = 0.99
    DISCOUNT_FACTOR = 0.99  # Training Hyperparameters section, gamma=0.99
    
    # Batch size: 512 transitions
    BATCH_SIZE = 512  # Training Hyperparameters section
    
    # Vector quantization commitment cost: β_vq = 0.25
    # Layman: This controls how strongly the encoder output is pulled toward codebook vectors
    VQ_COMMITMENT_COST = 0.25  # Training Hyperparameters section, beta_vq=0.25
    
    # Information bottleneck weight: λ2 = 0.01
    # Layman: This balances compression (lower MI with input) vs task performance
    IB_WEIGHT = 0.01  # Training Hyperparameters section, lambda_2=0.01
    
    # Communication penalty: α = 0.001
    # Layman: Small cost added each time an agent sends a message
    COMM_PENALTY_ALPHA = 0.001  # Training Hyperparameters section, alpha=0.001
    
    # Gating threshold: τ_gate = 0.5
    # Layman: Probability threshold above which an agent decides to communicate
    GATING_THRESHOLD = 0.5  # Training Hyperparameters section, tau_gate=0.5
    
    # Gumbel-Softmax temperature: τ = 1.0 (annealed to 0.1)
    # Layman: Temperature controls how "soft" vs "hard" the gating decision is
    GUMBEL_TEMP_START = 1.0  # Training Hyperparameters section
    GUMBEL_TEMP_END = 0.1  # Training Hyperparameters section
    
    # Codebook decay factor: γ = 0.99
    # Layman: Exponential moving average weight for updating codebook vectors
    CODEBOOK_DECAY = 0.99  # Training Hyperparameters section, gamma=0.99
    
    # Soft constraint penalty weight: λ_c = 0.01
    # Layman: Penalty multiplier when bandwidth budget is exceeded
    CONSTRAINT_PENALTY = 0.01  # Section III.B, lambda_c=0.01
    
    # ============================================================================
    # REWARD STRUCTURE (Section IV.A, Image 2, Equation 20)
    # ============================================================================
    
    # Target discovery reward
    REWARD_TARGET_DISCOVERY = 5.0  # Section IV.A reward structure
    
    # Target extraction reward
    REWARD_TARGET_EXTRACTION = 10.0  # Section IV.A reward structure
    
    # Coordination bonus for non-overlapping coverage
    REWARD_COORDINATION_BONUS = 2.0  # Section IV.A reward structure
    
    # Communication cost: α_comm = 0.1
    # Layman: Energy cost per message sent (like paying a toll)
    ALPHA_COMM = 0.1  # Equation 20, alpha_comm=0.1
    
    # Movement cost: α_move = 0.01
    # Layman: Energy cost per grid cell moved
    ALPHA_MOVE = 0.01  # Equation 20, alpha_move=0.01
    
    # ============================================================================
    # BANDWIDTH AND COMMUNICATION
    # ============================================================================
    
    # Bandwidth budget: B = 800 bits/episode (from Table I, our GVQ method)
    BANDWIDTH_BUDGET = 800  # Table I, Ours (GVQ) bits/episode
    
    # Bits per message: log2(K) = log2(16) = 4 bits
    # Layman: Since we have 16 possible messages, we need 4 bits to encode each
    BITS_PER_MESSAGE = 4  # Calculated from K=16 codebook size
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    
    # Number of training episodes
    NUM_EPISODES = 2000  # Typical for multi-agent RL, not explicitly stated
    
    # Maximum steps per episode
    MAX_EPISODE_STEPS = 100  # Typical for grid world tasks
    
    # Number of random seeds for statistical validation
    NUM_SEEDS = 8  # Section IV.C: "All experiments use 8 random seeds"
    
    # Early stopping patience
    EARLY_STOP_PATIENCE = 20  # Section IV.B: "20 episodes without improvement"
    
    # Logging frequency
    LOG_INTERVAL = 10  # Log every 10 episodes
    
    # ============================================================================
    # CONTEXT COMPONENTS FOR GATING (Section III.C)
    # ============================================================================
    
    # Enable/disable context components (all enabled by default per paper)
    USE_MESSAGE_HISTORY = True  # Section III.C context component
    USE_BANDWIDTH_UTILIZATION = True  # Section III.C context component
    USE_COORDINATION_REQUIREMENTS = True  # Section III.C context component
    USE_TEMPORAL_EFFICACY = True  # Section III.C context component
    
    # Message history length (recent messages to consider)
    MESSAGE_HISTORY_LENGTH = 5  # Reasonable default for temporal context
    
    # Temporal decay for message history weighting
    MESSAGE_HISTORY_DECAY = 0.9  # Exponential decay for older messages
    
    # ============================================================================
    # LOSS FUNCTION WEIGHTS (Equation 15)
    # ============================================================================
    
    # λ1 for VQ loss (coefficient for vector quantization loss)
    LAMBDA_1_VQ = 1.0  # Section III.F, lambda_1=1.0
    
    # λ2 for IB loss (information bottleneck weight)
    LAMBDA_2_IB = 0.01  # Section III.F, lambda_2=0.01
    
    # λ3 for gating loss (communication penalty)
    LAMBDA_3_GATE = 0.001  # Section III.F, lambda_3=0.001
    
    # ============================================================================
    # INFORMATION BOTTLENECK (Equation 18)
    # ============================================================================
    
    # Beta parameter for IB: controls compression vs prediction trade-off
    # Layman: Higher beta = care more about task performance, less about compression
    IB_BETA = 0.01  # Section IV.C main results, beta=0.01
    
    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use "cpu" if no GPU available
    
    # ============================================================================
    # DERIVED CONSTANTS (calculated from above parameters)
    # ============================================================================
    
    # Observation dimension: 5x5 local grid + agent features
    # Each cell has: empty(0)/obstacle(1)/target(2)/agent(3) = 4 possible states
    # Plus agent's own position, energy, etc.
    OBS_DIM = LOCAL_OBSERVATION_SIZE * LOCAL_OBSERVATION_SIZE + 10  # 25 grid + 10 features
    
    # Action space: 5 actions (stay, up, down, left, right)
    ACTION_DIM = 5
    
    # Maximum messages per episode given bandwidth budget
    # Layman: With 800 bits budget and 4 bits/message, we can send 200 messages max
    MAX_MESSAGES_PER_EPISODE = BANDWIDTH_BUDGET // BITS_PER_MESSAGE  # 800/4 = 200
    
    def __repr__(self):
        """Pretty print configuration"""
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_")])


# Create global config instance
cfg = Config()