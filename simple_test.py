"""
Simple test to verify the Active Inference implementation works
"""

import torch
from active_inference_ensemble import ActiveInferenceTokenEnsemble


class SimpleModel:
    """Mock model for testing"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def __call__(self, input_ids):
        # Return random logits
        batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
        seq_len = input_ids.shape[-1]
        return torch.randn(batch_size, seq_len, self.vocab_size)


def test_basic_functionality():
    print("🧠 Testing Active Inference Basic Functionality")
    
    # Setup
    vocab_size = 100
    models = [SimpleModel(vocab_size) for _ in range(3)]
    
    # Initialize ensemble
    ensemble = ActiveInferenceTokenEnsemble(
        models=models,
        vocab_size=vocab_size,
        sequence_length=50
    )
    
    print(f"✓ Created ensemble with {len(models)} models")
    
    # Test free energy computation
    beliefs = torch.rand(vocab_size)
    beliefs = beliefs / torch.sum(beliefs)
    
    priors = torch.rand(vocab_size) 
    priors = priors / torch.sum(priors)
    
    likelihood = torch.rand(vocab_size)
    likelihood = likelihood / torch.sum(likelihood)
    
    fe, complexity, accuracy = ensemble.compute_free_energy([1], beliefs, priors, likelihood)
    
    print(f"✓ Free Energy: {fe.item():.4f}")
    print(f"✓ Complexity: {complexity.item():.4f}")
    print(f"✓ Accuracy: {accuracy.item():.4f}")
    
    # Test belief updates
    initial_beliefs = ensemble.token_beliefs.clone()
    updated_beliefs = ensemble.minimize_free_energy_beliefs([5])
    
    print(f"✓ Beliefs updated (changed: {not torch.allclose(initial_beliefs, updated_beliefs)})")
    
    # Test prior updates
    context = torch.tensor([1, 2, 3])
    initial_prior = ensemble.transition_priors[3, 5].clone()
    ensemble.update_priors(context, 5)
    updated_prior = ensemble.transition_priors[3, 5]
    
    print(f"✓ Prior updated: {initial_prior.item():.6f} -> {updated_prior.item():.6f}")
    
    # Test token selection
    context = torch.tensor([1, 2, 3])
    token, efe, candidates = ensemble.active_inference_token_selection(context, top_k=5)
    
    print(f"✓ Selected token: {token.item()}, EFE: {efe:.4f}")
    print(f"✓ Considered {len(candidates)} candidates")
    
    # Test generation
    initial_tokens = torch.tensor([1, 2])
    generated = ensemble.generate_sequence(initial_tokens, max_length=5)
    
    print(f"✓ Generated sequence: {generated.tolist()}")
    
    # Test uncertainty metrics
    uncertainty = ensemble.get_uncertainty_metrics()
    print(f"✓ Uncertainty metrics: {uncertainty}")
    
    print("\n🎉 All basic functionality tests passed!")


if __name__ == "__main__":
    test_basic_functionality()