import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess_chords import SEQUENCE_LENGTH, MAPPING_PATH

class ChordsGenerator:
    """
    A class that wraps an LSTM model to generate melodies using chord sequences. 

    The class provides utilities for generating melodies based on a seed chord sequence,
    with additional control over generation length and variability (via temperature).
    """

    def __init__(self, model_path="./chords_model.keras"):
        """
        Initializes the ChordsGenerator by loading the pre-trained LSTM model 
        and mappings between chord symbols and integer encodings.

        Args:
            model_path (str): Path to the pre-trained Keras model. Default is './chords_model.keras'.

        Attributes:
            model_path (str): Path to the LSTM model file.
            model (keras.Model): Loaded LSTM model for melody generation.
            _mappings (dict): Mapping from chord symbols to integers.
            _inv_mappings (dict): Inverse mapping from integers to chord symbols.
            _start_symbols (list): List of start symbols used to initialize the sequence.
        """
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        self._start_symbols = ["/"] * SEQUENCE_LENGTH
        self._inv_mappings = {v: k for k, v in self._mappings.items()}

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """
        Generates a melody by iteratively predicting chords based on a seed sequence.

        Args:
            seed (str): Initial chord sequence in string format (e.g., "i bVII | bVI |").
            num_steps (int): Number of steps (chords) to generate.
            max_sequence_length (int): Maximum length of the input sequence for the model.
            temperature (float): Controls randomness in chord prediction. 
                                 Lower values result in less randomness, higher values increase variability.

        Returns:
            list: Generated melody as a list of chord symbols.
        """
        # Create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        # Map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]
        for _ in range(num_steps):
            # Limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            # Predict next chord
            probabilities = self.model.predict(np.array(seed).reshape(1, -1))[0]
            # Sample with temperature
            output_int = self._sample_with_temperature(probabilities, temperature)
            # Update seed
            seed.append(output_int)
            # Map integer to chord symbol
            output_symbol = self._inv_mappings[output_int]
            # Check whether we're at the end of a melody
            if output_symbol == "/":
                break
            # Update melody
            melody.append(output_symbol)
        return melody

    def _sample_with_temperature(self, probabilities, temperature):
        """
        Samples an index from a probability distribution with temperature adjustment.

        Args:
            probabilities (np.ndarray): Array of probabilities for each possible next chord.
            temperature (float): Value to control randomness in sampling. 
                                 Higher values lead to more exploration of possibilities.

        Returns:
            int: Index of the selected chord in the probability distribution.
        """
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)
        return index

if __name__ == "__main__":
    """
    Demonstrates usage of the ChordsGenerator class by generating a melody 
    from a seed sequence and printing it to the console.
    """
    mg = ChordsGenerator()
    seed = "i bVII | bVI |"
    melody = mg.generate_melody(seed, 30, SEQUENCE_LENGTH, 5.0)
    print(" ".join(melody))

