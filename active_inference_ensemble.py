import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


class ActiveInferenceTokenEnsemble:
    """
    Token-level Active Inference Ensemble for Language Models
    
    Implements the Free Energy Principle for token-level language generation:
    - Hierarchical priors P(s_t | s_{t-1}, position, context)
    - Free energy minimization for belief updates
    - Expected free energy for action selection
    - Variational message passing between token and model beliefs
    """
    
    def __init__(self, models, vocab_size, sequence_length=512, learning_rate=0.01):
        """
        Initialize Active Inference Ensemble
        
        Args:
            models: List of language models (e.g., GPT2LMHeadModel instances)
            vocab_size: Size of vocabulary
            sequence_length: Maximum sequence length for positional priors
            learning_rate: Learning rate for prior updates
        """
        self.models = models
        self.vocab_size = vocab_size
        self.seq_len = sequence_length
        self.lr = learning_rate
        
        # PRIORS: Hierarchical prior beliefs about token sequences
        # P(s_t | s_{t-1}) - transition priors
        self.transition_priors = torch.ones(vocab_size, vocab_size) / vocab_size
        
        # P(s_t | position) - positional priors  
        self.positional_priors = torch.ones(sequence_length, vocab_size) / vocab_size
        
        # P(model | context) - model selection priors
        self.model_priors = torch.ones(len(models)) / len(models)
        
        # Context-dependent priors cache
        self.context_priors_cache = {}
        
        # BELIEFS: Current posterior beliefs
        # q(s_t) - beliefs about hidden states (next token)
        self.token_beliefs = torch.ones(vocab_size) / vocab_size
        
        # q(model) - beliefs about which model is generating observations
        self.model_beliefs = torch.ones(len(models)) / len(models)
        
        # Precision parameters (inverse variance)
        self.precision_obs = 1.0  # Observation precision
        self.precision_trans = 1.0  # Transition precision
        
        # History tracking
        self.prediction_history = []
        self.free_energy_history = []
        
    def compute_free_energy(self, observations, beliefs, priors, likelihood):
        """
        Compute Free Energy: F = E_q[log q(s)] - E_q[log p(o,s)]
        
        Args:
            observations: Observed tokens
            beliefs: Current beliefs q(s)
            priors: Prior beliefs p(s)
            likelihood: Observation likelihood p(o|s)
            
        Returns:
            tuple: (free_energy, complexity, accuracy)
        """
        # Complexity cost: KL divergence between beliefs and priors
        complexity = torch.sum(beliefs * (torch.log(beliefs + 1e-8) - torch.log(priors + 1e-8)))
        
        # Accuracy: Expected log likelihood under beliefs
        accuracy = -torch.sum(beliefs * torch.log(likelihood + 1e-8))
        
        free_energy = complexity + accuracy
        return free_energy, complexity, accuracy
    
    def get_current_prior(self, context_tokens=None, position=None):
        """
        Compute hierarchical prior combining multiple sources
        P(s_t) = P(s_t | s_{t-1}) * P(s_t | position) * P(s_t | context_type)
        
        Args:
            context_tokens: Previous tokens for context
            position: Current position in sequence
            
        Returns:
            torch.Tensor: Combined prior distribution
        """
        if context_tokens is None or len(context_tokens) == 0:
            pos = 0 if position is None else min(position, self.seq_len - 1)
            return self.positional_priors[pos]
        
        # Transition prior: P(s_t | s_{t-1})
        last_token = context_tokens[-1].item()
        if last_token >= self.vocab_size:
            last_token = last_token % self.vocab_size
        transition_prior = self.transition_priors[last_token]
        
        # Positional prior: P(s_t | position)
        pos = len(context_tokens) if position is None else position
        pos = min(pos, self.seq_len - 1)
        positional_prior = self.positional_priors[pos]
        
        # Context-type prior: P(s_t | semantic_context)
        context_type_prior = self.get_context_type_prior(context_tokens)
        
        # Combine priors (assuming independence)
        combined_prior = (transition_prior * positional_prior * context_type_prior)
        combined_prior = combined_prior / torch.sum(combined_prior)  # Normalize
        
        return combined_prior
    
    def get_context_type_prior(self, context_tokens):
        """
        Learn context-dependent priors (e.g., mathematical, narrative, code)
        
        Args:
            context_tokens: Recent context tokens
            
        Returns:
            torch.Tensor: Context-specific prior distribution
        """
        if len(context_tokens) < 3:
            return torch.ones(self.vocab_size) / self.vocab_size
        
        # Create context signature from recent tokens
        context_signature = tuple(context_tokens[-5:].tolist()) if len(context_tokens) >= 5 else tuple(context_tokens.tolist())
        
        # Use cached context prior if available
        if context_signature in self.context_priors_cache:
            return self.context_priors_cache[context_signature]
        
        # Default to uniform prior for new contexts
        uniform_prior = torch.ones(self.vocab_size) / self.vocab_size
        self.context_priors_cache[context_signature] = uniform_prior.clone()
        
        return uniform_prior
    
    def compute_observation_likelihood(self, observations):
        """
        Compute P(o_t | s_t, models) - likelihood of observations given hidden states
        
        Args:
            observations: Observed tokens
            
        Returns:
            torch.Tensor: Observation likelihood distribution
        """
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations)
        
        likelihoods = torch.zeros(self.vocab_size)
        
        for token_id in range(self.vocab_size):
            likelihood = 0.0
            
            # Aggregate likelihood across models weighted by model beliefs
            for m_idx, model in enumerate(self.models):
                if token_id in observations:
                    # High likelihood for observed tokens
                    model_likelihood = 1.0
                else:
                    # Lower likelihood for unobserved tokens
                    model_likelihood = 0.1
                
                likelihood += self.model_beliefs[m_idx] * model_likelihood
            
            likelihoods[token_id] = likelihood
        
        return likelihoods / torch.sum(likelihoods)  # Normalize
    
    def minimize_free_energy_beliefs(self, observations, max_iterations=10):
        """
        Minimize free energy by updating beliefs via gradient descent
        This is the perception/inference step
        
        Args:
            observations: Observed tokens
            max_iterations: Maximum iterations for convergence
            
        Returns:
            torch.Tensor: Updated beliefs
        """
        for iteration in range(max_iterations):
            # Current free energy
            likelihood = self.compute_observation_likelihood(observations)
            current_prior = self.get_current_prior()
            
            F_old, _, _ = self.compute_free_energy(
                observations, self.token_beliefs, current_prior, likelihood
            )
            
            # Gradient of free energy w.r.t. beliefs
            # ∇_q F = log(q) - log(p) + ∇_q E_q[log p(o|s)]
            log_prior = torch.log(current_prior + 1e-8)
            log_likelihood = torch.log(likelihood + 1e-8)
            
            # Variational free energy gradient
            gradient = (torch.log(self.token_beliefs + 1e-8) - log_prior - 
                       self.precision_obs * log_likelihood)
            
            # Update beliefs (gradient descent)
            lr = 0.1
            self.token_beliefs = torch.softmax(
                torch.log(self.token_beliefs + 1e-8) - lr * gradient, dim=0
            )
            
            # Check convergence
            likelihood_new = self.compute_observation_likelihood(observations)
            F_new, _, _ = self.compute_free_energy(
                observations, self.token_beliefs, current_prior, likelihood_new
            )
            
            if abs(F_new - F_old) < 1e-6:
                break
        
        return self.token_beliefs
    
    def variational_message_passing(self, observations, context_tokens, num_iterations=5):
        """
        Implement message passing between belief nodes
        Updates beliefs about tokens AND models simultaneously
        
        Args:
            observations: Observed tokens
            context_tokens: Context for predictions
            num_iterations: Number of message passing iterations
            
        Returns:
            tuple: (updated_token_beliefs, updated_model_beliefs)
        """
        # Initialize messages
        token_to_model_msgs = torch.ones(self.vocab_size, len(self.models))
        model_to_token_msgs = torch.ones(len(self.models), self.vocab_size)
        
        for iteration in range(num_iterations):
            # UPDATE TOKEN BELIEFS
            # Message from models to tokens: how much each model supports each token
            for m_idx, model in enumerate(self.models):
                # Get model's prediction
                with torch.no_grad():
                    if len(context_tokens) > 0:
                        try:
                            model_logits = model(context_tokens.unsqueeze(0))[:, -1, :]
                            model_probs = F.softmax(model_logits, dim=-1).squeeze()
                        except:
                            # Fallback for model interface issues
                            model_probs = torch.ones(self.vocab_size) / self.vocab_size
                    else:
                        model_probs = torch.ones(self.vocab_size) / self.vocab_size
                
                # Message: P(token | model) * current model belief
                model_to_token_msgs[m_idx] = model_probs * self.model_beliefs[m_idx]
            
            # Update token beliefs by aggregating messages
            token_support = torch.sum(model_to_token_msgs, dim=0)  # Sum over models
            
            # Combine with priors
            current_prior = self.get_current_prior(context_tokens)
            self.token_beliefs = current_prior * token_support
            self.token_beliefs = self.token_beliefs / torch.sum(self.token_beliefs)
            
            # UPDATE MODEL BELIEFS  
            # Message from tokens to models: how well each model predicts likely tokens
            for t_idx in range(self.vocab_size):
                # How much do we believe this token will occur?
                token_prob = self.token_beliefs[t_idx]
                
                # How well does each model predict this token?
                for m_idx, model in enumerate(self.models):
                    if len(context_tokens) > 0:
                        try:
                            with torch.no_grad():
                                model_logits = model(context_tokens.unsqueeze(0))[:, -1, :]
                                model_token_prob = F.softmax(model_logits, dim=-1)[0, t_idx]
                        except:
                            model_token_prob = 1.0 / self.vocab_size
                    else:
                        model_token_prob = 1.0 / self.vocab_size
                    
                    # Message: P(model | token) weighted by token belief
                    token_to_model_msgs[t_idx, m_idx] = model_token_prob * token_prob
            
            # Update model beliefs
            model_support = torch.sum(token_to_model_msgs, dim=0)  # Sum over tokens
            self.model_beliefs = self.model_priors * model_support
            self.model_beliefs = self.model_beliefs / torch.sum(self.model_beliefs)
        
        return self.token_beliefs, self.model_beliefs
    
    def compute_expected_free_energy(self, context_tokens, action_token, horizon=3):
        """
        Compute Expected Free Energy for action selection
        G = Expected complexity + Expected accuracy - Pragmatic value
        
        Args:
            context_tokens: Current context
            action_token: Proposed next token
            horizon: Planning horizon
            
        Returns:
            float: Expected free energy
        """
        # Simulate taking this action
        extended_context = torch.cat([context_tokens, action_token.unsqueeze(0)])
        
        total_expected_free_energy = 0.0
        current_context = extended_context
        
        for step in range(horizon):
            # EXPECTED COMPLEXITY: KL[q(s_t+step) || p(s_t+step | context)]
            future_beliefs = self.predict_future_beliefs(current_context, step)
            future_prior = self.get_current_prior(current_context, len(current_context) + step)
            
            expected_complexity = torch.sum(
                future_beliefs * (torch.log(future_beliefs + 1e-8) - 
                                torch.log(future_prior + 1e-8))
            )
            
            # EXPECTED ACCURACY: E_q[-log p(o_t+step | s_t+step)]
            expected_observations = self.predict_observations(current_context, step)
            observation_likelihood = self.compute_observation_likelihood(expected_observations)
            
            expected_accuracy = -torch.sum(
                future_beliefs * torch.log(observation_likelihood + 1e-8)
            )
            
            # PRAGMATIC VALUE: Information gain and preference satisfaction
            current_entropy = -torch.sum(self.token_beliefs * torch.log(self.token_beliefs + 1e-8))
            future_entropy = -torch.sum(future_beliefs * torch.log(future_beliefs + 1e-8))
            epistemic_value = current_entropy - future_entropy  # Uncertainty reduction
            
            # Instrumental value (simplified)
            instrumental_value = torch.tensor(0.1)
            
            pragmatic_value = epistemic_value + instrumental_value
            
            # Total expected free energy for this time step
            step_expected_free_energy = (expected_complexity + expected_accuracy - 
                                       pragmatic_value)
            
            total_expected_free_energy += step_expected_free_energy
            
            # Predict next token for next iteration
            next_token = torch.multinomial(future_beliefs, 1)
            current_context = torch.cat([current_context, next_token])
        
        return total_expected_free_energy.item()
    
    def predict_future_beliefs(self, context_tokens, steps_ahead):
        """
        Predict what beliefs will be after taking steps_ahead actions
        
        Args:
            context_tokens: Current context
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            torch.Tensor: Predicted future beliefs
        """
        # Simplified: assume beliefs evolve toward ensemble prediction
        ensemble_prediction = torch.zeros(self.vocab_size)
        
        for i, model in enumerate(self.models):
            if len(context_tokens) > 0:
                try:
                    with torch.no_grad():
                        logits = model(context_tokens.unsqueeze(0))[:, -1, :]
                        probs = F.softmax(logits, dim=-1).squeeze()
                        ensemble_prediction += self.model_beliefs[i] * probs
                except:
                    # Fallback
                    ensemble_prediction += self.model_beliefs[i] / self.vocab_size
        
        # Blend current beliefs with ensemble prediction based on steps ahead
        blend_factor = min(steps_ahead * 0.3, 1.0)
        future_beliefs = (1 - blend_factor) * self.token_beliefs + blend_factor * ensemble_prediction
        return future_beliefs / torch.sum(future_beliefs)
    
    def predict_observations(self, context_tokens, steps_ahead):
        """
        Predict future observations (simplified)
        
        Args:
            context_tokens: Current context
            steps_ahead: Steps ahead to predict
            
        Returns:
            torch.Tensor: Predicted observations
        """
        # Simplified: sample from current beliefs
        predicted_token = torch.multinomial(self.token_beliefs, 1)
        return predicted_token
    
    def active_inference_token_selection(self, context_tokens, top_k=10):
        """
        Full active inference loop: perception -> action selection
        
        Args:
            context_tokens: Current context
            top_k: Number of top candidates to consider
            
        Returns:
            tuple: (selected_token, expected_free_energy, all_candidates)
        """
        # PERCEPTION: Update beliefs given current observations
        if len(context_tokens) > 0:
            last_observation = context_tokens[-1]
            self.minimize_free_energy_beliefs([last_observation])
            self.variational_message_passing([last_observation], context_tokens[:-1])
        
        # ACTION SELECTION: Choose action that minimizes expected free energy
        # Get candidate actions (top-k tokens from ensemble)
        candidate_logits = []
        for model in self.models:
            if len(context_tokens) > 0:
                try:
                    with torch.no_grad():
                        logits = model(context_tokens.unsqueeze(0))[:, -1, :]
                        candidate_logits.append(logits)
                except:
                    # Fallback
                    candidate_logits.append(torch.randn(1, self.vocab_size))
            else:
                candidate_logits.append(torch.randn(1, self.vocab_size))
        
        # Ensemble prediction weighted by model beliefs
        ensemble_logits = sum(
            self.model_beliefs[i] * logits for i, logits in enumerate(candidate_logits)
        )
        
        # Get top-k candidates
        top_k_values, top_k_indices = torch.topk(ensemble_logits.squeeze(), top_k)
        
        # Compute expected free energy for each candidate
        expected_free_energies = []
        for token_idx in top_k_indices:
            expected_fe = self.compute_expected_free_energy(
                context_tokens, token_idx, horizon=2
            )
            expected_free_energies.append((token_idx, expected_fe))
        
        # Select action with minimum expected free energy
        best_token, min_efe = min(expected_free_energies, key=lambda x: x[1])
        
        # Update priors based on selected action
        self.update_priors(context_tokens, best_token)
        
        # Store in history
        self.free_energy_history.append(min_efe)
        
        return best_token, min_efe, expected_free_energies
    
    def update_priors(self, context_tokens, observed_token):
        """
        Update priors based on observed sequences (online learning)
        
        Args:
            context_tokens: Previous context
            observed_token: Token that was observed/selected
        """
        if len(context_tokens) > 0:
            # Update transition priors
            last_token = context_tokens[-1].item()
            if last_token >= self.vocab_size:
                last_token = last_token % self.vocab_size
            if observed_token >= self.vocab_size:
                observed_token = observed_token % self.vocab_size
            
            # Exponential moving average update
            self.transition_priors[last_token] = (
                (1 - self.lr) * self.transition_priors[last_token] + 
                self.lr * F.one_hot(torch.tensor(observed_token), self.vocab_size).float()
            )
            
            # Normalize
            self.transition_priors[last_token] /= torch.sum(self.transition_priors[last_token])
        
        # Update positional priors
        position = len(context_tokens)
        if position < self.seq_len:
            self.positional_priors[position] = (
                (1 - self.lr) * self.positional_priors[position] + 
                self.lr * F.one_hot(torch.tensor(observed_token), self.vocab_size).float()
            )
            self.positional_priors[position] /= torch.sum(self.positional_priors[position])
        
        # Update context-specific priors
        if len(context_tokens) >= 3:
            context_signature = tuple(context_tokens[-5:].tolist()) if len(context_tokens) >= 5 else tuple(context_tokens.tolist())
            if context_signature in self.context_priors_cache:
                context_prior = self.context_priors_cache[context_signature]
                context_prior = (
                    (1 - self.lr) * context_prior + 
                    self.lr * F.one_hot(torch.tensor(observed_token), self.vocab_size).float()
                )
                self.context_priors_cache[context_signature] = context_prior / torch.sum(context_prior)
    
    def generate_sequence(self, initial_tokens, max_length=50, temperature=1.0):
        """
        Generate a sequence using active inference
        
        Args:
            initial_tokens: Starting tokens
            max_length: Maximum sequence length
            temperature: Sampling temperature
            
        Returns:
            torch.Tensor: Generated token sequence
        """
        current_tokens = initial_tokens.clone()
        
        for _ in range(max_length):
            # Active inference token selection
            next_token, efe, all_candidates = self.active_inference_token_selection(
                current_tokens, top_k=min(20, self.vocab_size)
            )
            
            # Add selected token
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0)])
            
            # Optional: stop on end token (would need to define end token)
            # if next_token == end_token_id:
            #     break
        
        return current_tokens
    
    def get_model_expertise_map(self, top_k=100):
        """
        Get which models are most trusted for which tokens
        
        Args:
            top_k: Number of top tokens to include
            
        Returns:
            dict: Mapping of token_id -> expert model info
        """
        # Get current model beliefs for different tokens
        expertise_map = {}
        
        # Sample different contexts to see model performance
        for token_id in range(min(top_k, self.vocab_size)):
            # Check which model has highest belief for this token
            # This is a simplified version - in practice you'd track performance over time
            model_scores = []
            
            for m_idx, model in enumerate(self.models):
                score = self.model_beliefs[m_idx].item()
                model_scores.append(score)
            
            best_model = np.argmax(model_scores)
            confidence = max(model_scores)
            
            if confidence > 1.0 / len(self.models):  # Above random
                expertise_map[token_id] = {
                    'expert_model': best_model,
                    'confidence': confidence,
                    'all_scores': model_scores
                }
        
        return expertise_map
    
    def get_uncertainty_metrics(self):
        """
        Get current uncertainty metrics
        
        Returns:
            dict: Various uncertainty measures
        """
        # Epistemic uncertainty (model disagreement)
        model_entropy = -torch.sum(self.model_beliefs * torch.log(self.model_beliefs + 1e-8))
        
        # Token belief entropy
        token_entropy = -torch.sum(self.token_beliefs * torch.log(self.token_beliefs + 1e-8))
        
        # Free energy history statistics
        if self.free_energy_history:
            avg_fe = np.mean(self.free_energy_history[-10:])  # Last 10 steps
            fe_trend = np.mean(np.diff(self.free_energy_history[-10:])) if len(self.free_energy_history) > 10 else 0
        else:
            avg_fe = 0
            fe_trend = 0
        
        return {
            'model_uncertainty': model_entropy.item(),
            'token_uncertainty': token_entropy.item(),
            'avg_free_energy': avg_fe,
            'free_energy_trend': fe_trend,
            'total_uncertainty': model_entropy.item() + token_entropy.item()
        }