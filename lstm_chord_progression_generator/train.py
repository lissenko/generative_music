import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from preprocess_chords import SINGLE_FILE_DATASET, SEQUENCE_LENGTH, create_data_generator, load

OUTPUT_UNITS = 206
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
SAVE_MODEL_PATH = "chords_model.keras"

def build_model(output_units, num_units, loss, learning_rate):
    """
    Builds and compiles an LSTM-based neural network for melody/chord generation.

    Args:
        output_units (int): Number of output units, typically the size of the vocabulary.
        num_units (list of int): List specifying the number of units in each LSTM layer.
        loss (str): Loss function for model training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        keras.Model: A compiled TensorFlow Keras model ready for training.
    """
    # Change input shape to match generator output
    input = keras.layers.Input(shape=(SEQUENCE_LENGTH,))
    # Add embedding layer to convert 1D input to 2D
    x = keras.layers.Embedding(output_units, 64, input_length=SEQUENCE_LENGTH)(input)
    x = keras.layers.LSTM(num_units[0])(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(output_units, activation="softmax")(x)
    model = keras.Model(input, output)
    
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    model.summary()
    return model

def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """
    Trains and saves an LSTM model for chord sequence generation.

    This function prepares the data generator, builds the model, trains it for a specified
    number of epochs, and saves the trained model to a file.

    Args:
        output_units (int, optional): Number of output units, typically the size of the vocabulary.
                                      Default is OUTPUT_UNITS.
        num_units (list of int, optional): List specifying the number of units in each LSTM layer.
                                           Default is NUM_UNITS.
        loss (str, optional): Loss function for model training. Default is LOSS.
        learning_rate (float, optional): Learning rate for the optimizer. Default is LEARNING_RATE.
    """
    # Create generator and determine steps per epoch
    generator = create_data_generator(SEQUENCE_LENGTH, BATCH_SIZE)
    
    # Estimate total number of sequences (adjust as needed)
    songs = load(SINGLE_FILE_DATASET)
    total_sequences = len(songs.split()) - SEQUENCE_LENGTH
    steps_per_epoch = total_sequences // BATCH_SIZE
    
    # Build and train model
    model = build_model(output_units, num_units, loss, learning_rate)
    model.fit(generator, 
              steps_per_epoch=steps_per_epoch, 
              epochs=EPOCHS)
    
    # Save the model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    """
    Executes the training pipeline for the LSTM model when the script is run directly.
    
    The `train` function is called to prepare the data, build the model, train it on the chord
    sequences, and save the trained model to the specified path.
    """
    train()

