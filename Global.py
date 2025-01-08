import tensorflow as tf

def initialize():
    global num_of_gen, num_of_ind
    num_of_gen = 20
    num_of_ind = 50
    
    global MRATE, ARATE, DIPRATE, LRATE
    MRATE = 0.1
    ARATE = 0.1
    DIPRATE = 0.2
    LRATE = 0.005
    
    global train_dataset, train_loader, test_dataset, test_loader
    global data, x, y, batch_size, pixelSize, num_of_chan,sequence_length, num_features
    data = []
    x = []
    y = []
    batch_size = 256
    pixelSize = 32
    num_of_chan = 3
    
    sequence_length = 10    
    num_features = 6        

    global num_of_input, num_of_output
    num_of_input = num_features #conv1d input num of channel
    num_of_output = 1 # output

    # GPU options
    global device, epoch
    gpus = tf.config.list_physical_devices('GPU')  # Get the list of available GPU devices
    if gpus:
        try:
            for gpu in gpus:
                # Enable dynamic memory allocation for GPUs
                tf.config.experimental.set_memory_growth(gpu, True)
            device = '/GPU:0'
            print("GPU available. Device in use:", device)
        except RuntimeError as e:
            print("Error during GPU setup:", e)
            device = '/CPU:0'
            print("Using CPU instead.")
    else:
        device = '/CPU:0'
        print("No GPU found. Using CPU.")

        
    epoch = 10
    
    
    global file1, file2
    file1 = None
    file2 = None


    # global searchingSpace
    # searchingSpace = ["Dense", "Conv1d", "LSTM", "PosEncode", "Transformer"]