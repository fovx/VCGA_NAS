# VCGA_NAS
Time series data automl using variable chromosome genetic algorithm 

We propose an enhanced Neural Architecture
Search (NAS) algorithm for time-series data using the Variable
Chromosome Genetic Algorithm (VCGA). VCGA introduces a
non-disjunction operation, enabling flexible evolution of network
structures. By restructuring the search space with self-attention
mechanisms and Transformer layers, the algorithm efficiently
explores architectures tailored for time-series tasks. Experiments
on NBA and ETTh1 datasets demonstrated the effectiveness of
the approach, with optimized architectures emerging from both
minimal and large initial models. Notably, the algorithm evolved a
novel architecture based on an encoder-only Transformer, where
the traditional positional encoding mechanism was replaced with
a Conv1D layer. This change alone significantly boosted the
performance of the existing Transformer model.

![overall process](https://github.com/user-attachments/assets/d8c2ca79-9257-4100-a478-85203b808b67)
