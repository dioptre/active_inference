# Active Inference Ensemble for LLMs

This repository implements a token-level active inference system for language model ensembles, based on the Free Energy Principle and active inference framework.

## Key Features

### 🧠 True Active Inference Components
- **Hierarchical Priors**: P(s_t | s_{t-1}, position, context)
- **Free Energy Minimization**: F = KL[q(s) || p(s)] - E_q[log p(o|s)]
- **Expected Free Energy**: For action selection with exploration/exploitation balance
- **Variational Message Passing**: Updates beliefs about both tokens and models

### 🔄 Token-Level Operation  
- Per-token belief tracking and updating
- Context-dependent prior learning
- Token-specific model expertise mapping
- Dynamic uncertainty quantification

### 🤖 Real Model Integration
- Works with actual GPT2 models (tested)
- Supports any transformer-based LLM
- Real-world data testing (20newsgroups dataset)

## Architecture

```
ActiveInferenceTokenEnsemble
├── Priors
│   ├── transition_priors: P(s_t | s_{t-1})
│   ├── positional_priors: P(s_t | position)  
│   └── model_priors: P(model | context)
├── Beliefs
│   ├── token_beliefs: q(s_t)
│   └── model_beliefs: q(model)
└── Inference
    ├── minimize_free_energy_beliefs()
    ├── variational_message_passing()
    └── compute_expected_free_energy()
```

## How It Works

1. **Perception Step**: Given observations, minimize free energy to update beliefs
2. **Action Step**: Select next token by minimizing expected free energy
3. **Learning Step**: Update priors based on observed transitions

### Free Energy Components
- **Complexity**: KL divergence between beliefs and priors (encourages staying close to priors)
- **Accuracy**: Negative log likelihood of observations (encourages accurate predictions)  
- **Pragmatic Value**: Information gain + preference satisfaction (drives exploration)

## Test Results

✅ **All tests pass** with real GPT2 models and real-world data:

- **Free Energy Computation**: Properly computes complexity + accuracy
- **Prior Updates**: Learns transition patterns from real text  
- **Belief Minimization**: Updates beliefs when observing new tokens
- **GPT2 Integration**: Successfully interfaces with transformers library
- **Hierarchical Priors**: Context-dependent prior computation
- **Full Pipeline**: End-to-end active inference on multiple text samples

## Usage

```python
# Setup ensemble with real models
models = [GPT2LMHeadModel.from_pretrained('gpt2') for _ in range(3)]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ensemble = ActiveInferenceTokenEnsemble(
    models=models, 
    vocab_size=tokenizer.vocab_size
)

# Generate with active inference
context = tokenizer.encode("The future of AI", return_tensors='pt')
for _ in range(50):
    # Perception: update beliefs given current context
    ensemble.minimize_free_energy_beliefs(context[-1:])
    
    # Action: select next token minimizing expected free energy
    next_token = ensemble.active_inference_token_selection(context)
    
    context = torch.cat([context, next_token.unsqueeze(0)])
```

## Key Advantages

1. **Principled Uncertainty**: True Bayesian uncertainty quantification
2. **Adaptive Learning**: Continuously updates priors from experience
3. **Exploration Balance**: Balances exploitation vs exploration via expected free energy  
4. **Token Specialization**: Different models can be experts for different tokens
5. **Context Sensitivity**: Priors adapt to mathematical, narrative, or other contexts

## Testing

Run the comprehensive test suite:

```bash
source .venv/bin/activate
python test_active_inference.py
```

Tests validate the system using:
- **Real GPT2 models** (not mocks)
- **Real text data** (20newsgroups dataset) 
- **Actual tokenization** and model inference
- **Mathematical validation** of active inference principles

The system successfully demonstrates token-level active inference with proper free energy minimization, hierarchical prior learning, and uncertainty-guided generation.