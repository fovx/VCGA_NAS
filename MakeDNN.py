import tensorflow as tf
import numpy as np
import copy
import Global
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense, Dropout, LayerNormalization

from tensorflow.keras.layers import Layer
import tensorflow as tf
import math
import time
    
import tensorflow as tf
import math

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model  # Used to dynamically verify input size

        # Compute positional encoding
        position = tf.range(0, max_len, dtype=tf.float32)[:, tf.newaxis]  # Shape: (max_len, 1)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model))
        
        # Even indices: sin, Odd indices: cos
        sin_part = tf.sin(position * div_term)
        cos_part = tf.cos(position * div_term)

        # Combine sin and cos results alternately
        pe = tf.concat([sin_part, cos_part], axis=-1)  # Shape: (max_len, d_model)

        # Add batch dimension
        self.pe = tf.expand_dims(pe, axis=0)  # Shape: (1, max_len, d_model)

    def call(self, x):
        # Check if the input dimension matches d_model
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Input dimension {x.shape[-1]} does not match d_model {self.d_model}")
        # Slice the positional encoding to match the input sequence length
        return x + self.pe[:, :tf.shape(x)[1], :]


class TransformerModel(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead, ffn_dim=64, dropout=0.1, num_layers=2):
        super(TransformerModel, self).__init__()

        self.num_layers = num_layers
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=nhead, 
            key_dim=d_model,
            dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_dim, activation='relu'),  # First layer of the Feed-Forward Network (FFN)
            tf.keras.layers.Dense(d_model)  # Match output dimensions to d_model
        ])
        
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        # Input tensor shape: (batch_size, seq_length, d_model)
        for _ in range(self.num_layers):
            # Multi-Head Attention
            attn_output = self.multi_head_attention(query=x, value=x, key=x)
            attn_output = self.dropout1(attn_output)
            out1 = self.layer_norm1(x + attn_output)  # Residual connection
            
            # Feed-Forward Network (FFN)
            ffn_output = self.ffn(out1)  # Apply FFN
            ffn_output = self.dropout2(ffn_output)
            x = self.layer_norm2(out1 + ffn_output)  # Residual connection
        
        return x  # Output tensor shape: (batch_size, seq_length, d_model)

class MakeDNN:
    def __init__(self, ind, param_weight=0.1, a=0.8):
       self.param_weight = param_weight
       self.a = a
       if ind is not None:
           self.ind = copy.deepcopy(ind)
           self.score = ind.score
           self.loss = 1.0
           self.accuracy = ind.accuracy
           self.dead = ind.dead
           
           with tf.device(Global.device):
               self.model = self.create_model()
               if self.model is not None and self.dead != 2:
                   self.compile_model()

    def check_link(self):
       """Connection Validation"""
       outcnt = 0
       if len(self.links) == 0:
           return False
        # Restricting nodes to not exceed two input nodes
       input_nodes = [n for n in self.ind.chromosomes if self.ind.chromosomes[n].isInput]
       if len(input_nodes) > 2:
            print(f"Invalid network: More than 2 input nodes detected. Count: {len(input_nodes)}")
            return False


       # Default Link Validation
       for l in self.links:
           # Check the existence of chromosomes specified in the link
           if l[0] not in self.ind.chromosomes or l[1] not in self.ind.chromosomes:
               return False
               
           # Check your connection to yourself
           if l[0] == l[1]:
               return False
               
           # Output Layer Count
           if self.ind.chromosomes[l[1]].isOutput:
               outcnt += 1
               
           # Check circular connection
           if [l[1], l[0]] in self.links:
               return False
               
           # Check the connection to the input
           if l[1] == 1:
               return False

       # Must have at least one output
       if outcnt == 0:
           return False

       # Examine the connection of all layers
       for c in self.ind.chromosomes:
           chrom = self.ind.chromosomes[c]
           if chrom.ctype != 0:  # About layers, not connections
               has_connection = False
               if not chrom.isOutput:  # If it's not the output layer
                   has_connection = any(l[0] == c for l in self.links)
               if not chrom.isInput and not chrom.isOutput:  
                   has_connection = has_connection or any(l[1] == c for l in self.links)
               if not has_connection and not (chrom.isInput or chrom.isOutput):
                   return False
                   
        # Restrict other layers from being connected behind output nodes
       for c in self.ind.chromosomes:
            chrom = self.ind.chromosomes[c]
            if chrom.isOutput:
                if any(l[0] == c for l in self.links):
                    print(f"Output node {c} is incorrectly connected to another layer.")
                    return False

       return True

    def check_model(self):
        """Model Structural Validation"""
        for l in self.links:
            start_chrom = self.ind.chromosomes[l[0]]
            end_chrom = self.ind.chromosomes[l[1]]

            # Validation of Dense Input Layer (type=1)
            if start_chrom.ctype == 1:  # Dense layer
                if len(start_chrom.genes) != 2:  # [seq_length, embedding_dim]
                    print(f"Invalid Dense layer genes: {start_chrom.genes}")
                    return False
                if start_chrom.genes[0] <= 1 or start_chrom.genes[1] <= 0:
                    print(f"Invalid Dense parameters: {start_chrom.genes}")
                    return False

            # Conv1D (type=2) Validation
            elif start_chrom.ctype == 2:  # Conv1D layer
                if len(start_chrom.genes) != 3:  # [filters, kernel_size, activation]
                    print(f"Invalid Conv1D genes: {start_chrom.genes}")
                    return False
                if start_chrom.genes[0] <= 1 or start_chrom.genes[1] <= 1:
                    print(f"Invalid Conv1D parameters: {start_chrom.genes}")
                    return False

            # LSTM (type=3) Validation
            elif start_chrom.ctype == 3:  # LSTM layer
                if len(start_chrom.genes) != 2:  # [units, return_sequences]
                    print(f"Invalid LSTM genes: {start_chrom.genes}")
                    return False
                if start_chrom.genes[0] <= 1 or start_chrom.genes[1] not in [0, 1]:
                    print(f"Invalid LSTM parameters: {start_chrom.genes}")
                    return False

            # Positional Encoding (ctype=4) Validation
            elif start_chrom.ctype == 4:  # Positional Encoding
                if len(start_chrom.genes) != 2: # [embedding_dim] = d_model
                    print(f"Invalid Positional Encoding genes: {start_chrom.genes}")
                    return False
                if start_chrom.genes[0] <= 1:
                    print(f"Invalid embedding_dim for Positional Encoding: {start_chrom.genes[0]}")
                    return False

            # Transformer (ctype=5) Validation
            elif start_chrom.ctype == 5:  # Transformer block
                if len(start_chrom.genes) != 4:  # [embedding_dim, num_heads, num_layers, activation]
                    print(f"Invalid Transformer genes: {start_chrom.genes}")
                    return False
                if start_chrom.genes[0] <= 1 or start_chrom.genes[1] <= 0 or start_chrom.genes[2] <= 1:
                    print(f"Invalid Transformer parameters: {start_chrom.genes}")
                    return False
                if start_chrom.genes[0] % start_chrom.genes[1] != 0:
                    print(f"Number of heads must divide embedding_dim: {start_chrom.genes}")
                    return False

        print("Model check passed successfully")
        return True

    @staticmethod
    def validate_and_correct_links(links, chromosomes):
        """
        A function that validates the link and corrects the abnormal connection
        """
        valid_links = []
        input_nodes = [n for n, c in chromosomes.items() if c.isInput]
        output_nodes = [n for n, c in chromosomes.items() if c.isOutput]

        # Filter all nodes to have at least one input/output
        for link in links:
            start, end = link
            if start in chromosomes and end in chromosomes:
                valid_links.append(link)

        # Correction of a path not connected from input to output
        def is_connected(start_nodes, target, links):
            visited = set()
            stack = list(start_nodes)
            while stack:
                node = stack.pop()
                if node == target:
                    return True
                if node not in visited:
                    visited.add(node)
                    stack.extend([end for s, end in links if s == node])
            return False

        for output in output_nodes:
            if not is_connected(input_nodes, output, valid_links):
                print(f"Output node {output} is not reachable. Attempting to correct.")
                # Add connection from one of the input layers to the output layer
                valid_links.append((input_nodes[0], output))

        # Remove duplicate links
        valid_links = list(set(tuple(link) for link in valid_links))

        # Remove a connectionless node
        connected_nodes = set(start for start, end in valid_links).union(set(end for start, end in valid_links))
        chromosomes = {k: v for k, v in chromosomes.items() if k in connected_nodes}

        return valid_links, chromosomes

    def create_model(self):
        """
        Modified create_model function
        """
        try:
            self.links = []
            for i in self.ind.chromosomes:
                if self.ind.chromosomes[i].ctype == 0:
                    self.links.append([
                        self.ind.chromosomes[i].genes[0],
                        self.ind.chromosomes[i].genes[1]
                    ])
            
            # Validation and Calibration
            self.links, self.ind.chromosomes = self.validate_and_correct_links(self.links, self.ind.chromosomes)

            if not self.check_link():
                print("Link check failed")
                self.dead = 2
                return None
            if not self.check_model():
                print("Model check failed")
                self.dead = 2
                return None
            
            self.dead = 1
            return self.build_model()
        except Exception as e:
            print(f"Error in create_model: {str(e)}")
            self.dead = 2
            return None

    def build_model(self):
        """모델 구축"""
        inputs = {}
        layers = {}
        sums = {}
        processed_layers = {}  # Track already processed layers
        prev_shapes = {}  # Save the output size for each layer

        layer_counter = {
            'conv1d': 0,
            'lstm': 0,
            'dense': 0,
            'positional_encoding': 0,
            'transformer': 0
        }

        def get_unique_layer_name(layer_type):
            layer_counter[layer_type] += 1
            return f"{layer_type}_{layer_counter[layer_type]}"

        # Create input layer
        for i in self.ind.chromosomes:
            if self.ind.chromosomes[i].isInput:
                inputs[str(i)] = tf.keras.layers.Input(
                    shape=(Global.sequence_length, Global.num_features),
                    name=f'input_{i}'
                )
                prev_shapes[i] = Global.num_features  # Set the initial input size
                layers[str(i)] = inputs[str(i)]
                processed_layers[str(i)] = inputs[str(i)]  # Save Input as Processed Layer
                print(f"Layer {i} (input): {prev_shapes[i]}")

                if self.ind.chromosomes[i].ctype == 1:  # Dense
                    x = tf.keras.layers.Flatten(name=f'flatten_{i}_input')(inputs[str(i)])
                    x = tf.keras.layers.Dense(
                        self.ind.chromosomes[i].genes[0],
                        name=get_unique_layer_name('dense')
                    )(x)
                    prev_shapes[i] = self.ind.chromosomes[i].genes[0]
                    print(f"Layer {i} (Dense): {prev_shapes[i]}")
                elif self.ind.chromosomes[i].ctype == 2:  # Conv1D
                    x = tf.keras.layers.Conv1D(
                        filters=self.ind.chromosomes[i].genes[0],
                        kernel_size=self.ind.chromosomes[i].genes[1],
                        padding='same',
                        activation='relu' if self.ind.chromosomes[i].genes[2] == 1 else 'sigmoid',
                        name=get_unique_layer_name('conv1d')
                    )(inputs[str(i)])
                    prev_shapes[i] = self.ind.chromosomes[i].genes[0]
                    print(f"Layer {i} (Conv1D): {prev_shapes[i]}")
                elif self.ind.chromosomes[i].ctype == 3:  # LSTM
                    x = tf.keras.layers.LSTM(
                        units=self.ind.chromosomes[i].genes[0],
                        activation='relu' if self.ind.chromosomes[i].genes[1] == 1 else 'sigmoid',
                        return_sequences=True,
                        name=get_unique_layer_name('lstm')
                    )(inputs[str(i)])
                    prev_shapes[i] = self.ind.chromosomes[i].genes[0]
                    print(f"Layer {i} (LSTM): {prev_shapes[i]}")
                elif self.ind.chromosomes[i].ctype == 4:  # Positional Encoding
                    x = PositionalEncoding(
                        d_model=prev_shapes[i],
                        name=get_unique_layer_name('positional_encoding')
                    )(inputs[str(i)])
                    prev_shapes[i] = prev_shapes[i]  # 유지
                    print(f"Layer {i} (Positional Encoding): {prev_shapes[i]}")
                elif self.ind.chromosomes[i].ctype == 5:  # Transformer
                    transformer_model = TransformerModel(
                        d_model=prev_shapes[i],
                        nhead=self.ind.chromosomes[i].genes[1],
                        num_layers=self.ind.chromosomes[i].genes[2]
                    )
                    x = transformer_model(inputs[str(i)])
                    prev_shapes[i] = self.ind.chromosomes[i].genes[0]
                    print(f"Layer {i} (Transformer): {prev_shapes[i]}")

                layers[str(i)] = x
                processed_layers[str(i)] = x
                sums[str(i)] = x

        def process_layer(x, chrom_id, prev_shape, is_output=False):
            chrom = self.ind.chromosomes[chrom_id]

            #For the Dense layer, check if Flatten is required
            if chrom.ctype == 1:
                # needs_flatten = len(x.shape) > 2  # 3D or higher inputs require Flatten
                
                # # Apply Flatten before Dense if required
                # if needs_flatten:
                #     x = tf.keras.layers.Flatten(name=f'flatten_{chrom_id}')(x)
                    
                x = tf.keras.layers.Dense(
                    units=chrom.genes[1] if not is_output else Global.num_of_output,
                    kernel_initializer='he_normal',
                    use_bias=True,
                    name=f'dense_{chrom_id}'
                )(x)

            # Conv1D layer
            elif chrom.ctype == 2:
                x = tf.keras.layers.Conv1D(
                    filters=chrom.genes[0],
                    kernel_size=chrom.genes[1], 
                    padding='same',
                    activation='relu',
                    name=f'conv1d_{chrom_id}'
                )(x)

            # LSTM layer
            elif chrom.ctype == 3:
                x = tf.keras.layers.LSTM(
                    units=chrom.genes[0],
                    activation='relu',
                    return_sequences=True, 
                    name=f'lstm_{chrom_id}'
                )(x)

            # Positional Encoding
            elif chrom.ctype == 4:
                d_model = prev_shape  # Use the output size of the previous layer
                x = PositionalEncoding(d_model=d_model)(x)

            # Transformer 레이어
            elif chrom.ctype == 5:
                d_model = prev_shape  # Use the output size of the previous layer
                if d_model % chrom.genes[1] != 0:  # Modify to divide into num_heads
                    print(f"d_model {d_model} is not divisible by num_heads {chrom.genes[1]}. Adjusting.")
                    d_model = chrom.genes[1] * (d_model // chrom.genes[1])
                transformer_model = TransformerModel(
                    d_model=d_model,
                    nhead=chrom.genes[1],
                    num_layers=chrom.genes[2]
                )
                x = transformer_model(x)

            return x

        def process_links(target_id):
            if str(target_id) in sums:
                return sums[str(target_id)]

            incoming_links = [l for l in self.links if l[1] == target_id]
            if not incoming_links:
                print(f"Warning: Could not process source layer {target_id}")
                return None

            source_id = incoming_links[0][0]
            source_output = process_links(source_id)

            if source_output is None:
                print(f"Warning: Source layer {source_id} for target {target_id} is None.")
                return None

            # Import prev_shape of the source layer
            source_prev_shape = prev_shapes.get(source_id)
            if source_prev_shape is None:
                raise ValueError(f"Source layer {source_id} does not have a defined `prev_shape`.")

            # Processing the current layer (add prev_shape)
            processed = process_layer(source_output, target_id, prev_shape=source_prev_shape)
            prev_shapes[target_id] = processed.shape[-1]  #Update the prev_shape of the current layer
            sums[str(target_id)] = processed
            return processed


        outputs = []
        for i in self.ind.chromosomes:
            if self.ind.chromosomes[i].isOutput:
                out = process_links(i)
                if out is None:
                    raise ValueError(f"Output layer {i} could not be processed.")
                outputs.append(out)

        if len(outputs) > 1:
            final_output = tf.keras.layers.Add()(outputs)
        else:
            final_output = outputs[0]

        # # Check for additional Flatten prior to final output
        # if len(final_output.shape) > 2:
        #     final_output = tf.keras.layers.Flatten(name='final_flatten')(final_output)
            
        model = tf.keras.Model(
            inputs=list(inputs.values())[0] if len(inputs) == 1 else list(inputs.values()),
            outputs=final_output
        )
        print("\nModel Architecture:")
        model.summary()
        return model
        
    def compile_model(self):
       """Compiling models for time series prediction"""
       self.model.compile(
           optimizer=tf.keras.optimizers.Adam(
               learning_rate=Global.LRATE
           ),
           loss='mse',
           metrics=['mae']
       )

    def set_default_scores(self):
       """Default Score Settings"""
       self.loss = 1.0
       self.accuracy = 0.0
       self.score = 0.0
       self.ind.loss = 1.0
       self.ind.accuracy = 0.0
       self.ind.score = 0.0 

    def calculate_score(self):
        """New Score Calculation Logic"""
        total_params = self.model.count_params()
        params_MB = total_params * 4 / (1024 * 1024)  # Converting to parameter size (MB)
        self.score = 1 - (self.a * self.loss + (1 - self.a) * params_MB * self.param_weight)
        print(f"Score calculated: {self.score:.6f} (Loss: {self.loss}, Params: {params_MB:.2f} MB)")

    def train_and_evaluate(self):
        """Record learning and evaluation performance and execution time"""
        if self.model is None or self.dead == 2:
            self.set_default_scores()
            return

        try:
            start_time = time.time()  # Generation run time starts
            x_train, y_train = [], []
            for x_batch, y_batch in Global.train_loader:
                x_train.append(x_batch.numpy())
                y_train.append(y_batch.numpy())
            x_train = np.concatenate(x_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)

            x_test, y_test = [], []
            for x_batch, y_batch in Global.test_loader:
                x_test.append(x_batch.numpy())
                y_test.append(y_batch.numpy())
            x_test = np.concatenate(x_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            history = self.model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                batch_size=Global.batch_size,
                epochs=Global.epoch,
                verbose=0
            )

            self.loss = history.history['val_loss'][-1]
            self.accuracy = 1.0 - history.history['val_mae'][-1]

            self.ind.loss = self.loss
            self.ind.accuracy = self.accuracy

            self.calculate_score()  # Calculate new scores
            self.ind.score = self.score  # Reflect score to ind object

            end_time = time.time()  # End Generation Run Time
            print(f"Training completed in {end_time - start_time:.2f} seconds")
            print(f"Loss: {self.loss:.6f}, Score: {self.score:.6f}")

        except Exception as e:
            print(f"Error during training: {e}")
            self.set_default_scores()
