import tensorflow as tf
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import os
import copy
from GenePool import GenePool
from CrossBreed import CrossBreed
import Global
from MakeDNN import MakeDNN

def openFile():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="choice file",
        filetypes=(("excel file","*.xlsx"),("all files","*.*"))
    )
    npath = filepath

    Global.data = pd.read_excel(npath)
    
    # 텐서 변환
    inputs = [
        Global.data['1st'].values,
        Global.data['2nd'].values,
        Global.data['3rd'].values,
        Global.data['4th'].values
    ]
    inputs = [tf.cast(x, tf.float32) for x in inputs]
    
    # Stack inputs
    with tf.device(Global.device):
        Global.x = tf.stack(inputs, axis=0)
        Global.x = tf.transpose(Global.x)
        Global.y = tf.cast(Global.data['result'].values, tf.float32)

class GADNN:
    def __init__(self, num_of_gen, num_of_ind):
        self.num_of_gen = num_of_gen
        self.num_of_ind = num_of_ind

        # Create an initial pool (use GenePool classes as they are)
        self.GP = GenePool(self.num_of_gen, self.num_of_ind)
        self.CB = CrossBreed()

        # Log Directory Settings
        epath = os.getcwd()
        if 'GAML' not in epath:
            epath = os.path.join(epath, 'GAML')
        try:
            os.makedirs(os.path.join(epath, 'Log'), exist_ok=True)
        except OSError:
            print("Failed to create directory!!!!!")
            raise
        
        epath = os.path.join(epath, 'Log')
        os.chdir(epath)

        # Creating a Timestamp Directory
        now = datetime.now()
        nowdate = now.strftime('%m%d')
        nowtime = now.strftime('%H%M%S')
        npath = os.path.join(epath, f'Log({now.year%100}{nowdate}.{nowtime})')
        os.makedirs(npath)
        os.chdir(npath)

        # Initialize log files
        with open('./Result.log', 'a') as Global.file2:
            Global.file2.write(f'///비분리율:\t{Global.ARATE}\t,돌연변이율:\t{Global.MRATE}\t,염색체의존도:\t{Global.DIPRATE}\t,학습율:\t{Global.LRATE}\n')
            Global.file2.write(f'///입력 개수:\t{Global.num_of_input}\t,출력 개수:\t{Global.num_of_output}\n\n')

        # generational evolution
        for j in range(self.num_of_gen):
            print(f'\nGenerartion {j}:')
            
            # average calculation
            total_chrom = sum(ind.num_of_chrom for ind in self.GP.individuals)
            total_layer = sum(ind.num_of_layer for ind in self.GP.individuals)
            avg_chrom = total_chrom/Global.num_of_ind
            avg_layer = total_layer/Global.num_of_ind

            # Evaluate each object
            for i, ind in enumerate(self.GP.individuals):
                # Log by Generation
                with open(f'./G{j}.log', 'a') as Global.file1:
                    Global.file1.write(f'\nInd {i}\n')
                    for k in ind.chromosomes:
                        Global.file1.write(f'C{k}\t{ind.chromosomes[k]}\n')

                # Evaluate objects
                loss, accuracy = self.play(ind)
                score = accuracy

                # Record results
                with open(f'./G{j}.log', 'a') as Global.file1:
                    Global.file1.write(f'{ind}\n')
                    text = f'Individual{i}\tLoss : {loss:.6f}    ,Accuracy : {accuracy:.6f}    ,Score : {ind.score:.6f}    ,num_of_chrom : {ind.num_of_chrom}    ,num_of_layer : {ind.num_of_layer}    ,num_of_link : {ind.num_of_link}'
                    print(text)
                    Global.file1.write(f'Loss :\t{loss:.6f}\n')

            # Sort and record results
            self.GP.sort1()
            with open('./Result.log', 'a') as Global.file2:
                Global.file2.write(f'///Generation {j}\t,best loss:\t{self.GP.individuals[0].loss:.6f}\t,best accuracy:\t{self.GP.individuals[0].accuracy:.6f}\t,best score:\t{self.GP.individuals[0].score:.6f}\t,best의 염색체 개수:\t{self.GP.individuals[0].num_of_chrom}\t,best의 레이어 개수:\t{self.GP.individuals[0].num_of_layer}\t,best의 링크 개수:\t{self.GP.individuals[0].num_of_link}\n')

            if j == Global.num_of_gen-1:
                break

            # crossbreed (PyTorch 버전의 선택/교배 로직 유지)
            childs = []
            # The top 20% are the same
            childs.extend([self.GP.selectN(i) for i in range(self.num_of_ind//5)])
            # Top 5% interbreeding with top objects
            childs.extend([self.CB.breed(self.GP.selectBest(), self.GP.selectGood()) 
                         for _ in range(self.num_of_ind//5)])
            # Top 5% crossings with secondary parent objects
            childs.extend([self.CB.breed(self.GP.selectN(1), self.GP.selectGood()) 
                         for _ in range(self.num_of_ind//5)])
            # Top 5% crossbreeding
            childs.extend([self.CB.breed(self.GP.selectGood(), self.GP.selectGood()) 
                         for _ in range(self.num_of_ind//5)])
            # Best and bottom 5% crossings
            childs.extend([self.CB.breed(self.GP.selectBest(), self.GP.selectBad()) 
                         for _ in range(self.num_of_ind//10)])
            # Cross the top 5% and the bottom 5%
            childs.extend([self.CB.breed(self.GP.selectGood(), self.GP.selectBad()) 
                         for _ in range(self.num_of_ind//10)])
            
            self.GP.individuals = copy.deepcopy(childs)

    def play(self, ind):
        """개체 평가"""
        print(f"\nPlaying individual with {ind.num_of_chrom} chromosomes")
        try:
            with tf.device(Global.device):
                MD = MakeDNN(ind, param_weight=0.001, a=0.5)  # passing the score calculation method parameters
                
                if not hasattr(MD, 'model') or MD.dead == 2:
                    print("Model is dead or invalid")
                    ind.loss = 1.0
                    ind.accuracy = 0.0
                    ind.score = 0.0
                    return 1.0, 0.0
                else:
                    # Converting Model Summary to Text
                    model_summary = []
                    MD.model.summary(print_fn=lambda x: model_summary.append(x))
                    model_summary = "\n".join(model_summary)

                    # Log Model Summary
                    with open(f'./G{self.num_of_gen}.log', 'a') as Global.file1:
                        Global.file1.write("\n=== Model Summary ===\n")
                        Global.file1.write(model_summary + "\n")
                        Global.file1.write("=====================\n")
                    
                    # Model Learning and Evaluation
                    MD.train_and_evaluate()
                    
                    # Update scores for Individual objects
                    ind.loss = MD.loss
                    ind.accuracy = MD.accuracy
                    ind.score = MD.score

                    del MD
                    
                    print(f"Training completed - Loss: {ind.loss:.6f}, Accuracy: {ind.accuracy:.6f}, Score: {ind.score:.6f}")
                    return ind.loss, ind.accuracy

        except Exception as e:
            print(f"Error in play: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            ind.loss = 1.0
            ind.accuracy = 0.0
            ind.score = 0.0
            return 1.0, 0.0
def main():
    gasim = GADNN(Global.num_of_gen, Global.num_of_ind)
    for i in range(10):
    #     # Output only objects that exist
    # for i in range(min(10, len(gasim.GP.individuals))):
        print(gasim.GP.individuals[i])

    # Data Normalization with MinMax Scaling
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    # Initialize Global Variables
    Global.initialize()
    
    # Load Time Series Data
    data_path = 'C:/Users/xorjf/Desktop/AtomicGene/merged_nabdata.csv'  # Path where Excel data is saved as CSV
    data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

    # Feature and prediction target settings
    feature_columns = ['art_load_balancer_spikes', 'art_increase_spike_density', 'art_daily_nojump', 'art_daily_jumpsup', 'art_daily_jumpsdown', 'art_daily_flatmiddle']
    target_column = 'art_load_balancer_spikes'  # Prediction target

    # Data Normalization with MinMax Scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])  # All feature scaling

    # Time series sequence generation function
    def create_sequences(data, target, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])      # input sequence
            y.append(target[i + seq_length])      # Predicted value (Next time value)
        return np.array(X), np.array(y)

    # Sequence generation
    seq_length = Global.sequence_length  # Lookback Length
    X, y = create_sequences(scaled_features, data[target_column].values, seq_length)

    # Check data shape
    print("X shape (input sequences):", X.shape)  # (samples, sequence_length, features)
    print("y shape (target values):", y.shape)   # (samples,)

    # Split Learning/Test Data
    train_size = int(len(X) * 0.8)
    train_x, train_y = X[:train_size], y[:train_size]
    test_x, test_y = X[train_size:], y[train_size:]

    # Creating a TensorFlow Data Loader
    with tf.device(Global.device):
        Global.train_loader = tf.data.Dataset.from_tensor_slices((train_x, train_y))\
            .shuffle(buffer_size=1024)\
            .batch(Global.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

        Global.test_loader = tf.data.Dataset.from_tensor_slices((test_x, test_y))\
            .batch(Global.batch_size)\
            .prefetch(tf.data.AUTOTUNE)

        print(f"Using device: {Global.device}")

    # Check data shape
    for x_batch, y_batch in Global.train_loader.take(1):
        print("Train Input Batch Shape:", x_batch.shape)  # 예상: (batch_size, sequence_length, features)
        print("Train Label Batch Shape:", y_batch.shape)  # 예상: (batch_size,)

    # Running the GADNN Model
    main()