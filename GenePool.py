from Individual import Individual
import random

class GenePool:
    number_of_individuals = 0
    number_of_gen = 0
    individuals = []

    def __init__(self, number_of_gen, number_of_individuals):
        self.number_of_individuals = 0
        self.number_of_gen = 0
        self.individuals = []

        self.number_of_individuals = number_of_individuals
        self.number_of_gen = number_of_gen

        for i in range(self.number_of_individuals):
            self.individuals.append(Individual(hn = 1))
            #self.individuals[i].score = random.random()

    def __str__(self):
        return "number_of_individuals = %d, number_of_gen = %d"%(self.number_of_individuals, self.number_of_gen)
        
    def sort1(self):
        self.individuals.sort(key=lambda x: x.score, reverse=True) #점수로 내림차순 정렬

    def selectN(self, n): #n등 개체 선택
        return self.individuals[n]

    def selectBest(self): #0등 개체 선택
        return self.individuals[0]

    def selectGood(self):
        r = random.randint(0, self.number_of_individuals//20) #상위 5%
        return self.selectN(r)

    def selectBad(self):
        r = random.randint(self.number_of_individuals*95//100, self.number_of_individuals-1) #하위 5%
        return self.selectN(r)  