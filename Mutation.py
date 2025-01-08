import Global
import random
from Chromosome import Chromosome
from AtomicGene import AtomicGene
import copy

# Function to change the type of layer in the chromosome
def layerChange(c1):
    chrom = copy.deepcopy(c1)  # Deep copy of the input chromosome
    new_chrom = copy.deepcopy(chrom)  # Create a new copy of the chromosome
    n = random.randint(1, 6)  # Randomly select a layer type

    if n == 1:  # Dense layer
        new_chrom.ctype = 1
        new_chrom.genes = [chrom.genes[0], 1]
    elif n == 2:  # Conv1D layer
        new_chrom.ctype = 2
        new_chrom.genes = [chrom.genes[0], 3, 1]
    elif n == 3:  # LSTM layer
        new_chrom.ctype = 3
        new_chrom.genes = [chrom.genes[0], 1]  # [hidden_units, activations]
    elif n == 4:  # Positional Encoding
        new_chrom.ctype = 4
        new_chrom.genes = [64, 1]  # [embedding_dim, placeholder activation]
    elif n == 5:  # Transformer
        new_chrom.ctype = 5
        new_chrom.genes = [32, 4, 128, 1]  # [d_model, num_heads, ffn_dim, activation_function]

    return new_chrom

# Function to exchange or modify genes in the chromosome
def exchange(c1):
    chrom = copy.deepcopy(c1)  # Deep copy of the input chromosome
    
    chrom_len = len(chrom.genes)  # Get the length of the genes list
    r = random.randint(0, chrom_len)  # Select a random gene to modify
    r2 = random.randint(0, 4)  # Select a random operation to apply

    # Protect the output layer from modifications
    if chrom.isOutput:  # If it's an output layer, return it unchanged
        return chrom
    
    # Modify the activation function if it's the last gene
    if r == chrom_len - 1:  # Activation (last element in genes list)
        chrom.genes[r] = random.randint(0, 2)
    # Change the layer type if selected index equals the length of the genes list
    elif r == chrom_len:
        chrom = layerChange(chrom)
    else:
        # Apply transformations to the selected gene
        if r2 == 0:
            chrom.genes[r] = chrom.genes[r] - 1
        elif r2 == 1:
            chrom.genes[r] = chrom.genes[r] + 1
        elif r2 == 2:
            chrom.genes[r] = chrom.genes[r] // 2
        elif r2 == 3:
            chrom.genes[r] = chrom.genes[r] * 2
        elif r2 == 4:
            chrom.genes[r] = chrom.genes[r] * 3 // 2
    
    return chrom

# Function to determine whether a mutation occurs
def isMutation():
    r = random.randrange(10000)  # Generate a random number between 0 and 9999
    return r < Global.MRATE * 10000  # Mutation occurs if the random number is less than mutation rate
