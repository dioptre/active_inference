"""
Example usage of Active Inference Ensemble with GPT2
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from active_inference_ensemble import ActiveInferenceTokenEnsemble


def main():
    print("🧠 Active Inference Ensemble Example")
    
    # Setup models and tokenizer
    print("Loading GPT2 models...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create ensemble of GPT2 models (in practice, these could be different model variants)
    models = []
    for i in range(3):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        models.append(model)
        print(f"✓ Loaded model {i+1}")
    
    # Initialize Active Inference Ensemble
    ensemble = ActiveInferenceTokenEnsemble(
        models=models,
        vocab_size=tokenizer.vocab_size,
        sequence_length=128,
        learning_rate=0.01
    )
    
    print(f"✓ Initialized ensemble with {len(models)} models and {tokenizer.vocab_size:,} vocab size")
    
    # Example 1: Basic generation
    print("\n📝 Example 1: Basic Active Inference Generation")
    prompt = "The future of artificial intelligence"
    input_tokens = tokenizer.encode(prompt, return_tensors='pt').squeeze()
    
    print(f"Prompt: '{prompt}'")
    print("Generating with active inference...")
    
    # Generate sequence
    generated_tokens = ensemble.generate_sequence(
        initial_tokens=input_tokens,
        max_length=20
    )
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated: '{generated_text}'")
    
    # Example 2: Step-by-step active inference
    print("\n🔍 Example 2: Step-by-step Active Inference Process")
    context = tokenizer.encode("Machine learning is", return_tensors='pt').squeeze()
    
    print(f"Starting context: '{tokenizer.decode(context)}'")
    
    for step in range(5):
        print(f"\n--- Step {step+1} ---")
        
        # Get uncertainty metrics before selection
        uncertainty_before = ensemble.get_uncertainty_metrics()
        print(f"Uncertainty before: {uncertainty_before['total_uncertainty']:.4f}")
        
        # Select next token with active inference
        next_token, expected_fe, all_candidates = ensemble.active_inference_token_selection(
            context, top_k=5
        )
        
        # Show candidates and their expected free energies
        print("Candidate tokens and their expected free energies:")
        for token_id, efe in all_candidates:
            token_text = tokenizer.decode([token_id])
            print(f"  '{token_text}' -> EFE: {efe:.4f}")
        
        # Add selected token
        selected_text = tokenizer.decode([next_token])
        print(f"Selected: '{selected_text}' (EFE: {expected_fe:.4f})")
        
        context = torch.cat([context, next_token.unsqueeze(0)])
        current_text = tokenizer.decode(context)
        print(f"Current text: '{current_text}'")
        
        # Get uncertainty after selection
        uncertainty_after = ensemble.get_uncertainty_metrics()
        print(f"Uncertainty after: {uncertainty_after['total_uncertainty']:.4f}")
    
    # Example 3: Learning and adaptation
    print("\n🎯 Example 3: Prior Learning from Real Text")
    
    # Feed some training text to update priors
    training_texts = [
        "The cat sat on the mat",
        "Machine learning algorithms require large datasets",
        "Python is a programming language",
        "Neural networks are inspired by biological neurons"
    ]
    
    print("Training on example texts to update priors...")
    
    for i, text in enumerate(training_texts):
        tokens = tokenizer.encode(text, return_tensors='pt').squeeze()
        
        # Process each token to update priors
        for j in range(1, len(tokens)):
            context = tokens[:j]
            target = tokens[j]
            
            # Update beliefs and priors
            ensemble.minimize_free_energy_beliefs([target])
            ensemble.update_priors(context, target)
        
        print(f"✓ Processed text {i+1}: '{text}'")
    
    # Show learned transition patterns
    print("\nLearned some transition patterns:")
    test_contexts = ["the", "machine", "neural"]
    
    for context_word in test_contexts:
        try:
            context_id = tokenizer.encode(context_word)[0]
            if context_id < ensemble.vocab_size:
                # Get top transitions for this context
                transitions = ensemble.transition_priors[context_id]
                top_indices = torch.topk(transitions, 3).indices
                
                print(f"After '{context_word}':")
                for idx in top_indices:
                    next_word = tokenizer.decode([idx])
                    prob = transitions[idx].item()
                    print(f"  -> '{next_word}' (prob: {prob:.4f})")
        except:
            print(f"  -> Could not process '{context_word}'")
    
    # Example 4: Model expertise analysis
    print("\n🏆 Example 4: Model Expertise Analysis")
    
    expertise_map = ensemble.get_model_expertise_map(top_k=20)
    
    print("Model expertise for different tokens:")
    count = 0
    for token_id, info in expertise_map.items():
        if count < 5:  # Show first 5
            token_text = tokenizer.decode([token_id])
            expert_model = info['expert_model']
            confidence = info['confidence']
            print(f"Token '{token_text}' -> Expert: Model {expert_model} (confidence: {confidence:.3f})")
            count += 1
    
    # Example 5: Uncertainty tracking during generation
    print("\n📊 Example 5: Uncertainty-Guided Generation")
    
    prompt = "Quantum computing"
    context = tokenizer.encode(prompt, return_tensors='pt').squeeze()
    
    print(f"Generating with uncertainty tracking from: '{prompt}'")
    
    uncertainty_history = []
    text_history = [prompt]
    
    for step in range(8):
        # Get current uncertainty
        uncertainty = ensemble.get_uncertainty_metrics()
        uncertainty_history.append(uncertainty['total_uncertainty'])
        
        # Generate next token
        next_token, efe, _ = ensemble.active_inference_token_selection(context)
        context = torch.cat([context, next_token.unsqueeze(0)])
        
        current_text = tokenizer.decode(context)
        text_history.append(current_text)
        
        print(f"Step {step+1}: Uncertainty={uncertainty['total_uncertainty']:.3f}, EFE={efe:.3f}")
        print(f"  Text: '{current_text}'")
    
    print("\n✅ Active Inference Ensemble Demo Complete!")
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"- Generated {len(uncertainty_history)} tokens with active inference")
    print(f"- Average uncertainty: {sum(uncertainty_history)/len(uncertainty_history):.3f}")
    print(f"- Final text: '{text_history[-1]}'")
    print(f"- Models adapted priors based on {len(training_texts)} training examples")


if __name__ == "__main__":
    main()