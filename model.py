"""
Neural network architectures for GVQ method
Based on Farooq & Iqbal (2026) - arXiv:2602.02035v1

Implements:
- Policy Network (3-layer MLP, 256 hidden units)
- Encoder Network (2-layer MLP, 128 hidden units → 64-dim)
- Gating Network (2-layer MLP, 64 hidden units)
- Decoder Network (2-layer MLP, 128 hidden units)
- Vector Quantization Codebook (K=16 vectors, dim=64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


class PolicyNetwork(nn.Module):
    """
    Policy network: 3-layer MLP with 256 hidden units and ReLU activation
    (Section IV.C - Implementation Details)
    
    Layman: This is the "brain" that decides what action each agent should take
    based on what it observes (like a chess player deciding their next move).
    """
    
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        # Input: observation + received messages
        input_dim = cfg.OBS_DIM + cfg.CODEBOOK_DIM  # obs + message embedding
        
        # 3-layer MLP with 256 hidden units
        self.fc1 = nn.Linear(input_dim, cfg.POLICY_HIDDEN_DIM)  # 256 hidden units
        self.fc2 = nn.Linear(cfg.POLICY_HIDDEN_DIM, cfg.POLICY_HIDDEN_DIM)
        self.fc3 = nn.Linear(cfg.POLICY_HIDDEN_DIM, cfg.POLICY_HIDDEN_DIM)
        
        # Output: action logits (5 actions) and value estimate
        self.action_head = nn.Linear(cfg.POLICY_HIDDEN_DIM, cfg.ACTION_DIM)
        self.value_head = nn.Linear(cfg.POLICY_HIDDEN_DIM, 1)
        
        # Hidden state for RNN-like behavior (for context)
        self.hidden_dim = cfg.POLICY_HIDDEN_DIM
        
    def forward(self, obs, received_message, hidden_state=None):
        """
        Forward pass through policy network
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            received_message: Embedded message from other agents [batch, codebook_dim]
            hidden_state: Previous hidden state [batch, hidden_dim]
        
        Returns:
            action_logits: Action probabilities [batch, action_dim]
            value: State value estimate [batch, 1]
            new_hidden: Updated hidden state [batch, hidden_dim]
        """
        # Concatenate observation and received message
        x = torch.cat([obs, received_message], dim=-1)
        
        # 3-layer MLP with ReLU activation (as specified in paper)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # This becomes our hidden state
        
        # Action and value outputs
        action_logits = self.action_head(x)
        value = self.value_head(x)
        
        return action_logits, value, x  # x is the new hidden state


class EncoderNetwork(nn.Module):
    """
    Encoder network: 2-layer MLP with 128 hidden units mapping to 64-dim
    (Section IV.C - Implementation Details)
    
    Layman: This compresses the agent's observation into a compact "summary"
    that can be turned into a message (like summarizing a story into key points).
    """
    
    def __init__(self):
        super(EncoderNetwork, self).__init__()
        
        # Input: observation + policy hidden state
        input_dim = cfg.OBS_DIM + cfg.POLICY_HIDDEN_DIM
        
        # 2-layer MLP with 128 hidden units → 64-dimensional output
        self.fc1 = nn.Linear(input_dim, cfg.ENCODER_HIDDEN_DIM)  # 128 hidden units
        self.fc2 = nn.Linear(cfg.ENCODER_HIDDEN_DIM, cfg.ENCODER_OUTPUT_DIM)  # 64-dim output
        
    def forward(self, obs, hidden_state):
        """
        Encode observation into continuous latent representation
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            hidden_state: Policy network hidden state [batch, policy_hidden_dim]
        
        Returns:
            z: Continuous latent representation [batch, 64]
        """
        x = torch.cat([obs, hidden_state], dim=-1)
        
        # 2-layer MLP with ReLU
        x = F.relu(self.fc1(x))
        z = self.fc2(x)  # 64-dimensional latent representation
        
        return z


class GatingNetwork(nn.Module):
    """
    Gating network: 2-layer MLP with 64 hidden units and sigmoid output
    (Section IV.C - Implementation Details)
    
    Layman: This decides "should I send a message now?" based on the situation
    (like deciding whether to call a teammate or stay silent).
    """
    
    def __init__(self):
        super(GatingNetwork, self).__init__()
        
        # Input: observation + policy hidden state + communication context
        # Context includes: message history, bandwidth utilization, 
        # coordination requirements, temporal efficacy
        context_dim = 4  # 4 context components (Section III.C)
        input_dim = cfg.OBS_DIM + cfg.POLICY_HIDDEN_DIM + context_dim
        
        # 2-layer MLP with 64 hidden units → sigmoid output
        self.fc1 = nn.Linear(input_dim, cfg.GATING_HIDDEN_DIM)  # 64 hidden units
        self.fc2 = nn.Linear(cfg.GATING_HIDDEN_DIM, 1)  # Single output (probability)
        
    def forward(self, obs, hidden_state, context):
        """
        Compute gating probability (whether to communicate)
        
        Args:
            obs: Observation tensor [batch, obs_dim]
            hidden_state: Policy hidden state [batch, policy_hidden_dim]
            context: Communication context [batch, 4]
                - Message history
                - Bandwidth utilization
                - Coordination requirements
                - Temporal efficacy
        
        Returns:
            gate_prob: Probability of communicating [batch, 1]
        """
        x = torch.cat([obs, hidden_state, context], dim=-1)
        
        # 2-layer MLP with ReLU and sigmoid
        x = F.relu(self.fc1(x))
        gate_logit = self.fc2(x)
        gate_prob = torch.sigmoid(gate_logit)  # Sigmoid output (Equation 9)
        
        return gate_prob


class VectorQuantizer(nn.Module):
    """
    Vector Quantization module with learned codebook
    (Section III.D - Vector Quantized Message Encoding)
    
    Codebook size: K=16 vectors of dimension 64
    
    Layman: This is like having a dictionary of 16 pre-approved messages.
    Instead of sending any message, we find the closest match from our dictionary.
    This is similar to texting with emoji - limited choices but very efficient.
    """
    
    def __init__(self):
        super(VectorQuantizer, self).__init__()
        
        # Codebook: K=16 vectors of dimension 64 (Implementation Details section)
        self.num_embeddings = cfg.CODEBOOK_SIZE  # K = 16
        self.embedding_dim = cfg.CODEBOOK_DIM  # dim = 64
        
        # Initialize codebook with random vectors
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 
                                           1.0 / self.num_embeddings)
        
        # Commitment cost: β_vq = 0.25 (Training Hyperparameters section)
        self.commitment_cost = cfg.VQ_COMMITMENT_COST  # beta_vq = 0.25
        
        # Exponential moving average for codebook updates
        # Decay factor: γ = 0.99 (Training Hyperparameters section)
        self.decay = cfg.CODEBOOK_DECAY  # gamma = 0.99
        
        # Codebook usage tracking (for health analysis, Figure 4)
        self.register_buffer('cluster_size', torch.zeros(self.num_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
    def forward(self, z):
        """
        Quantize continuous latent z to discrete codebook vector
        
        Args:
            z: Continuous latent representation [batch, 64]
        
        Returns:
            quantized: Discrete quantized vector [batch, 64]
            encoding_indices: Codebook indices [batch] (this is the "message")
            vq_loss: Vector quantization loss
        """
        # Flatten if needed
        z_flattened = z.view(-1, self.embedding_dim)  # [batch, 64]
        
        # Calculate distances to codebook vectors (Equation 12)
        # Layman: Find which dictionary message is closest to what we want to say
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        
        # Find nearest codebook vector (arg min distance)
        encoding_indices = torch.argmin(distances, dim=1)  # [batch]
        
        # Quantize: replace z with nearest codebook vector
        quantized = self.embedding(encoding_indices)  # [batch, 64]
        
        # Compute VQ loss (Equation 17)
        # Layman: This loss has two parts:
        # 1. Pull the codebook vector toward z (so codebook learns)
        # 2. Pull z toward the codebook vector (so encoder learns to use codebook)
        
        # Commitment loss: ||z - sg[quantized]||^2
        commitment_loss = F.mse_loss(z, quantized.detach())
        
        # Embedding loss: ||sg[z] - quantized||^2
        embedding_loss = F.mse_loss(z.detach(), quantized)
        
        # Combined VQ loss (Equation 17)
        vq_loss = embedding_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from quantized to z
        # Layman: This is a trick to let gradients flow through the non-differentiable
        # quantization operation (like a magical gradient bridge)
        quantized = z + (quantized - z).detach()
        
        # Update codebook with exponential moving average (Equations 13-14)
        if self.training:
            self._update_codebook_ema(z_flattened, encoding_indices)
        
        return quantized, encoding_indices, vq_loss
    
    def _update_codebook_ema(self, z, encoding_indices):
        """
        Update codebook using exponential moving average (EMA)
        Implements Equations 13-14 from paper
        
        Args:
            z: Continuous latents [batch, 64]
            encoding_indices: Selected codebook indices [batch]
        """
        # One-hot encode indices
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Update cluster sizes (Equation 13): N_k = γN_k + (1-γ)∑1[m_i = c_k]
        self.cluster_size = self.cluster_size * self.decay + \
                           (1 - self.decay) * torch.sum(encodings, dim=0)
        
        # Update running sum of embeddings
        dw = torch.matmul(encodings.t(), z)  # [K, 64]
        self.embed_avg = self.embed_avg * self.decay + (1 - self.decay) * dw
        
        # Update codebook vectors (Equation 14): c_k = γc_k + (1-γ)(∑z_i)/N_k
        n = torch.sum(self.cluster_size)
        cluster_size = (
            (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        )
        
        self.embedding.weight.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))


class DecoderNetwork(nn.Module):
    """
    Decoder network: 2-layer MLP with 128 hidden units
    (Section IV.C - Implementation Details)
    
    Layman: This takes a received message (codebook vector) and processes it
    into a form the policy network can understand (like translating a telegram).
    """
    
    def __init__(self):
        super(DecoderNetwork, self).__init__()
        
        # Input: quantized message vector (64-dim)
        # Output: processed message representation (64-dim)
        
        # 2-layer MLP with 128 hidden units
        self.fc1 = nn.Linear(cfg.CODEBOOK_DIM, cfg.DECODER_HIDDEN_DIM)  # 128 hidden
        self.fc2 = nn.Linear(cfg.DECODER_HIDDEN_DIM, cfg.CODEBOOK_DIM)  # 64-dim output
        
    def forward(self, quantized_message):
        """
        Process received quantized message
        
        Args:
            quantized_message: Quantized vector [batch, 64]
        
        Returns:
            processed_message: Processed representation [batch, 64]
        """
        # 2-layer MLP with ReLU
        x = F.relu(self.fc1(quantized_message))
        processed_message = self.fc2(x)
        
        return processed_message


class GVQAgent(nn.Module):
    """
    Complete GVQ (Gated Vector Quantization) agent
    Combines all components: Policy, Encoder, Gating, VQ, Decoder
    
    Layman: This is the complete "AI agent" that can observe, decide, act,
    and communicate efficiently with teammates.
    """
    
    def __init__(self):
        super(GVQAgent, self).__init__()
        
        # Initialize all networks
        self.policy_net = PolicyNetwork()
        self.encoder_net = EncoderNetwork()
        self.gating_net = GatingNetwork()
        self.vector_quantizer = VectorQuantizer()
        self.decoder_net = DecoderNetwork()
        
        # Gumbel-Softmax temperature for training (annealed from 1.0 to 0.1)
        self.gumbel_temp = cfg.GUMBEL_TEMP_START  # τ = 1.0 initially
        
    def forward(self, obs, hidden_state, received_messages, context, 
                training=True):
        """
        Full forward pass through GVQ agent
        
        Args:
            obs: Observation [batch, obs_dim]
            hidden_state: Previous hidden state [batch, policy_hidden_dim]
            received_messages: Messages from other agents [batch, codebook_dim]
            context: Communication context [batch, 4]
            training: Whether in training mode
        
        Returns:
            action_logits: Action probabilities [batch, action_dim]
            value: State value [batch, 1]
            new_hidden: Updated hidden state
            comm_prob: Communication probability [batch, 1]
            message_token: Message token index [batch] (0-15)
            quantized_message: Quantized message vector [batch, 64]
            vq_loss: Vector quantization loss
        """
        # Step 1: Policy network decides action
        action_logits, value, new_hidden = self.policy_net(
            obs, received_messages, hidden_state
        )
        
        # Step 2: Gating network decides whether to communicate (Equation 9)
        gate_prob = self.gating_net(obs, new_hidden, context)
        
        # Step 3: If communicating, encode message
        # Encode observation to continuous latent (Equation 11)
        z = self.encoder_net(obs, new_hidden)
        
        # Quantize to discrete message (Equation 12)
        quantized_message, message_token, vq_loss = self.vector_quantizer(z)
        
        # Step 4: Use Gumbel-Softmax for differentiable gating (Equation 10)
        if training:
            # Gumbel-Softmax trick for soft gating
            # Layman: This lets us train with probabilities but behave like
            # hard decisions (communicate yes/no)
            gumbel_gate = self._gumbel_softmax_sample(gate_prob, self.gumbel_temp)
        else:
            # Hard gating at test time
            gumbel_gate = (gate_prob > cfg.GATING_THRESHOLD).float()
        
        return (action_logits, value, new_hidden, gate_prob, gumbel_gate,
                message_token, quantized_message, vq_loss)
    
    def _gumbel_softmax_sample(self, logits, temperature):
        """
        Gumbel-Softmax sampling for differentiable discrete decisions
        Implements Equation 10 from paper
        
        Layman: This is a mathematical trick that lets us make discrete yes/no
        decisions during training while still being able to use gradient descent.
        """
        # Sample Gumbel noise
        gumbel_noise_1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        gumbel_noise_0 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        
        # Apply Gumbel-Softmax (Equation 10)
        y_1 = torch.exp((torch.log(logits + 1e-20) + gumbel_noise_1) / temperature)
        y_0 = torch.exp(gumbel_noise_0 / temperature)
        
        gumbel_gate = y_1 / (y_1 + y_0)
        
        return gumbel_gate
    
    def anneal_temperature(self, step, total_steps):
        """
        Anneal Gumbel-Softmax temperature from 1.0 to 0.1
        (Training Hyperparameters: τ = 1.0 annealed to 0.1)
        
        Layman: Start with "soft" decisions early in training, gradually make
        them more "crisp" as the agent learns.
        """
        progress = step / total_steps
        self.gumbel_temp = cfg.GUMBEL_TEMP_START * (1 - progress) + \
                          cfg.GUMBEL_TEMP_END * progress


def compute_context(agent_id, message_history, bandwidth_used, 
                     targets_remaining, recent_rewards):
    """
    Compute communication context for gating network (Section III.C)
    
    Context components:
    1. Message History: recent messages from other agents
    2. Bandwidth Utilization: current usage relative to budget
    3. Coordination Requirements: estimated need based on task progress
    4. Temporal Communication Efficacy: historical effectiveness
    
    Args:
        agent_id: Current agent index
        message_history: List of recent messages
        bandwidth_used: Current bandwidth consumption
        targets_remaining: Number of uncollected targets
        recent_rewards: Recent reward improvements
    
    Returns:
        context: Context vector [4]
    """
    # Component 1: Message History (with temporal decay)
    # Layman: How many messages have we received recently?
    if cfg.USE_MESSAGE_HISTORY and len(message_history) > 0:
        # Weight recent messages more heavily with exponential decay
        weights = [cfg.MESSAGE_HISTORY_DECAY ** i 
                  for i in range(len(message_history))]
        message_history_feature = sum(weights) / cfg.MESSAGE_HISTORY_LENGTH
    else:
        message_history_feature = 0.0
    
    # Component 2: Bandwidth Utilization
    # Layman: How much of our communication budget have we used?
    if cfg.USE_BANDWIDTH_UTILIZATION:
        bandwidth_util = bandwidth_used / cfg.BANDWIDTH_BUDGET  # Normalized [0,1]
    else:
        bandwidth_util = 0.0
    
    # Component 3: Coordination Requirements
    # Layman: How much do we need to coordinate right now?
    if cfg.USE_COORDINATION_REQUIREMENTS:
        # Higher coordination need when many targets remain
        coord_requirement = targets_remaining / cfg.MAX_TARGETS
    else:
        coord_requirement = 0.0
    
    # Component 4: Temporal Communication Efficacy
    # Layman: Has communicating helped us succeed recently?
    if cfg.USE_TEMPORAL_EFFICACY and len(recent_rewards) > 0:
        # Average recent reward improvements
        temporal_efficacy = np.mean(recent_rewards) if recent_rewards else 0.0
    else:
        temporal_efficacy = 0.0
    
    # Combine into context vector
    context = torch.FloatTensor([
        message_history_feature,
        bandwidth_util,
        coord_requirement,
        temporal_efficacy
    ]).unsqueeze(0).to(cfg.DEVICE)  # [1, 4]
    
    return context