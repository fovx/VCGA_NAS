from Chromosome import Chromosome
import Global
import copy

class Individual:
    score = 0
    loss = 0
    accuracy = 0
    dead = 0
    num_of_chrom = 0
    num_of_layer = 0
    num_of_link = 0
    chromosomes = {}
    
    def __init__(self, chromosomes=None, hn=None):  #init model define
        self.score = 0
        self.loss = 0
        self.num_of_chrom = 0
        self.num_of_layer = 0
        self.num_of_link = 0
        self.chromosomes = {}

        if chromosomes is not None:
            self.chromosomes = copy.deepcopy(chromosomes)
        
        if hn is not None:
            # Conv1D  (ctype=2)
            self.chromosomes[1] = Chromosome(ctype=2,genes=[32, 3, 1], i=True) #filter, kernal, activation
            # Dense (ctype=1)
            self.chromosomes[2] = Chromosome(ctype=1,genes=[Global.num_of_output ,1],o=True)

            # LSTM  (ctype=3)
            # self.chromosomes[3] = Chromosome(ctype=3,genes=[64, 1],)# [hidden_units, activation]
            
            #positionalencoding
            # self.chromosomes[3] = Chromosome(ctype=4,genes=[64, 1],)# [hidden_units, activation]
            #Transformer
            self.chromosomes[4] = Chromosome(ctype=5,genes=[64, 8, 256, 1],)# [hidden_units, activation]
            
            # connect layer(link)
            self.chromosomes[5] = Chromosome(ctype=0, genes=[1, 4])
            self.chromosomes[6] = Chromosome(ctype=0, genes=[4, 2]) 
            # self.chromosomes[7] = Chromosome(ctype=0, genes=[4, 2]) 

        # calculate layer and link 
        self.num_of_chrom = len(self.chromosomes)
        for c in self.chromosomes:
            if self.chromosomes[c].ctype == 0:
                self.num_of_link += 1
            else:
                self.num_of_layer += 1

    def __str__(self):
        return "loss =\t%.4lf\t, accuracy =\t%.4lf\t,num_of_chrom =\t%d\t,num_of_layer =\t%d\t,num_of_link =\t%d" % (
            self.loss, self.accuracy, self.num_of_chrom, self.num_of_layer, self.num_of_link
        )

    def AddChromosome(self, chrom):
        i = 3
        while i in self.chromosomes:
            i += 1
        self.chromosomes[i] = chrom
        self.num_of_chrom += 1
        return i

    def setScore(self, n):
        """점수 설정 - accuracy 기반"""
        self.score = self.accuracy