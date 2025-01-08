from AtomicGene import AtomicGene

class Chromosome:
    ctype = 0 #-1은 잘못된 염색체
    isInput = False #입력 레이어인지?
    isOutput = False #출력 레이어인지?
    genes = []

    def __init__(self, ctype = None, genes = None, i = False, o = False) : ##genes are list [nodes, activations]
                                                                            #convolution is [filter, kernel_x, kernel_y, Maxpool size, activations]
                                                                            ##[start_layer, target_layer, activations]
        self.ctype = 0 
        self.genes = []
        if i:
            self.isInput = True
        if o:
            self.isOutput = True
        if ctype is not None:
            self.ctype = ctype
        if genes is not None:
            self.genes = genes
        #self.genes.append(AtomicGene(1))
    
    def AddGene(self, t1) :
        self.genes.append(AtomicGene(t1))

    def __eq__(self, other) : 
        if(self.genes == other.genes) :
            return True
        else :
            return False
    def __str__(self):
        return "ctype =\t%d\t,\tgenes =\t%s"%(self.ctype, self.genes)
    
    def printChro(self) :
        print(self.genes)