# Report — Task 03: Text Generation with Markov Chains

## Objective
Implement a Markov chain text generator that models the probability of the next word given the previous n words.

## How It Works

### Markov Chains
A Markov chain is a stochastic model where the next state depends only on the current state (Markov property). For text generation:
- **State** = a tuple of `n` consecutive words (an n-gram)
- **Transition** = the set of words observed to follow that n-gram in training data

### Training
The `train()` method scans the corpus word-by-word, building a `dict[tuple → list[str]]`. Each key is an n-gram; the value is all words that followed it.

### Generation
Starting from a random (or seeded) n-gram, the model repeatedly samples a successor from the transition list, appends it, and slides the window forward.

### Order (n)
| Order | Behaviour |
|-------|-----------|
| 1 | Random, creative, often incoherent |
| 2 | Readable phrases, mild repetition |
| 3 | Near-grammatical, less variety |
| 4 | Very faithful to training corpus |

## Implementation Highlights
- **Zero external ML libraries** — pure Python `random` + `collections`
- Single `MarkovChain` class with `train()`, `generate()`, `stats()` methods
- Flask REST endpoint `/generate` keeps the model stateless per request
- UI built in vanilla JS — no framework overhead

## Results
With the default 15-sentence corpus and order=2, the model produces grammatically plausible sentences that blend topics from the training set. Higher orders yield more coherent but less novel output.
