import Global
from Chromosome import Chromosome
from Individual import Individual
import Mutation
import random
import copy

class CrossBreed:
    def __init__(self, num_of_ind=None):
        if num_of_ind is not None:
            self.num_of_ind = num_of_ind
    
    def breed(self, p1, p2):
        size = max(p1.num_of_chrom, p2.num_of_chrom)
        
        seedInd = Individual()
        feedInd = Individual()
        r = random.randint(0, 1)
        if r == 0:
            seedInd = copy.deepcopy(p1)
            feedInd = copy.deepcopy(p2)
        else:
            seedInd = copy.deepcopy(p2)
            feedInd = copy.deepcopy(p1)

        # Dictionary to track the output size of the previous layer
        prev_shapes = {1: Global.num_features}  # Initial output size of chromosome input layer 1
        
        for c in feedInd.chromosomes:
            if feedInd.chromosomes[c].ctype == -1:
                continue
            isAtc = self.isAttach()
            r = random.randint(0, 1)
            
            if c in seedInd.chromosomes:
                if not isAtc:
                    if r == 0:  # 50% chance of inheritance of feedInd's chromosomes
                        if Mutation.isMutation():
                            seedInd.chromosomes[c] = Mutation.exchange(feedInd.chromosomes[c])
                        else:
                            seedInd.chromosomes[c] = copy.deepcopy(feedInd.chromosomes[c])
                    elif r == 1:
                        if Mutation.isMutation():
                            seedInd.chromosomes[c] = Mutation.exchange(seedInd.chromosomes[c])
                else:  # Non-separation
                    ar = random.randint(0, 1)  # Select non-separation type
                    if ar == 0:  # add
                        newChrom = copy.deepcopy(feedInd.chromosomes[c])
                        if c == 1 or c == 2:  # If it's chromosome 1 or 2, make it not input/output
                            newChrom.isInput = False
                            newChrom.isOutput = False
                        if newChrom.ctype != 0:  # if it's a layer, it mutates once
                            # Transfer `layerChange` to `prev_shape`
                            newChrom = Mutation.layerChange(newChrom)
                        newi = seedInd.AddChromosome(newChrom)

                        # Update connection information by adding a new layer
                        if newChrom.ctype != 0:
                            newChrom2 = copy.deepcopy(Chromosome(ctype=0, genes=[c, newi]))
                            seedInd.AddChromosome(newChrom2)
                            for c2 in seedInd.chromosomes:
                                if seedInd.chromosomes[c2].ctype == 0 and seedInd.chromosomes[c2].genes[0] == c and seedInd.chromosomes[c2].genes[1] != newi:
                                    seedInd.chromosomes[c2].genes[0] = newi
                            for c2 in feedInd.chromosomes:
                                if feedInd.chromosomes[c2].ctype == 0 and feedInd.chromosomes[c2].genes[0] == c and feedInd.chromosomes[c2].genes[1] != newi:
                                    feedInd.chromosomes[c2].genes[0] = newi
                    elif ar == 1:  # concubine
                        if c != 1 and c != 2:  # If it's chromosome 1 or 2, don't separate (because it's input and output)
                            dropChrom = seedInd.chromosomes.pop(c)
                            if dropChrom.ctype != 0:  # Delete all relevant links if it is a layer
                                tmpl = 1
                                while True:
                                    stopFlag = True
                                    for c2 in seedInd.chromosomes:
                                        if seedInd.chromosomes[c2].ctype == 0:
                                            if seedInd.chromosomes[c2].genes[1] == c:
                                                tmpl = seedInd.chromosomes[c2].genes[0]
                                                seedInd.chromosomes.pop(c2)
                                                stopFlag = False
                                                break
                                    if stopFlag:
                                        break
                                for c2 in seedInd.chromosomes:
                                    if seedInd.chromosomes[c2].ctype == 0:
                                        if seedInd.chromosomes[c2].genes[0] == c:
                                            seedInd.chromosomes[c2].genes[0] = tmpl
                                tmpl = 1
                                while True:
                                    stopFlag = True
                                    for c2 in feedInd.chromosomes:
                                        if feedInd.chromosomes[c2].ctype == 0:
                                            if feedInd.chromosomes[c2].genes[1] == c:
                                                tmpl = feedInd.chromosomes[c2].genes[0]
                                                feedInd.chromosomes[c2].ctype = -1
                                                stopFlag = False
                                                break
                                    if stopFlag:
                                        break
                                for c2 in feedInd.chromosomes:
                                    if feedInd.chromosomes[c2].ctype == 0:
                                        if feedInd.chromosomes[c2].genes[0] == c:
                                            feedInd.chromosomes[c2].genes[0] = tmpl

        seedInd.num_of_chrom = len(seedInd.chromosomes)
        seedInd.num_of_layer = 0
        seedInd.num_of_link = 0
        for c in seedInd.chromosomes:
            if seedInd.chromosomes[c].ctype == 0:
                seedInd.num_of_link += 1
            else:
                seedInd.num_of_layer += 1
        return seedInd

    def isAttach(self):
        r = random.randrange(10000)
        if r < Global.ARATE*10000:
            return True
        else :
            return False
