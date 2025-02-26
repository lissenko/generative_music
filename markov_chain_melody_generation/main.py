import random
from music21 import stream, note, midi

from preprocessing import preprocessing, SILENCE

class MarkovChain:
    """
    A Markov Chain model for generating music sequences.

    Attributes:
        order (int): The order of the Markov Chain.
        observations (dict): The observed transitions between contexts and targets.
        alphabet (list): The set of unique targets in the training data.
        order_zero_model (MarkovChain): A zero-order Markov Chain for fallback and initial sampling.
    """

    def __init__(self, order, use_order_zero=True):
        """
        Initialize a Markov Chain model.

        Parameters:
            order (int): The order of the Markov Chain.
            use_order_zero (bool): Whether to use a zero-order Markov Chain as fallback.
        """
        self.order = order
        self.observations = {}
        self.alphabet = []
        if use_order_zero:
            self.order_zero_model = MarkovChain(0, False)

    def train(self, data):
        """
        Train the Markov Chain on the given data.

        Parameters:
            data (list): A list of sequences to train on.
        """
        self._train_order_zero_model(data)
        self._train_on_data(data)

    def _train_order_zero_model(self, data):
        """
        Train the zero-order Markov Chain on the given data.

        Parameters:
            data (list): A list of sequences to train on.
        """
        self.order_zero_model._train_on_data(data)

    def _train_on_data(self, data):
        """
        Train the Markov Chain of the specified order on the given data.

        Parameters:
            data (list): A list of sequences to train on.
        """
        order = self.order
        for sequence in data:
            size_seq = len(sequence)
            for j in range(order, size_seq):
                target = sequence[j]
                if target not in self.alphabet:
                    self.alphabet.append(target)
                context = tuple([sequence[j - k] for k in range(order, 0, -1)])
                if context not in self.observations:
                    self.observations[context] = {target: 1}
                else:
                    if target not in self.observations[context]:
                        self.observations[context][target] = 1
                    else:
                        self.observations[context][target] += 1

    def _count_context(self, context):
        """
        Count the total occurrences of a context in the observations.

        Parameters:
            context (tuple): The context to count.

        Returns:
            int: The total count of the context.
        """
        count = 0
        if context in self.observations:
            count = sum(self.observations[context].values())
        return count

    def _count_context_target(self, context, target):
        """
        Count the occurrences of a specific target following a context.

        Parameters:
            context (tuple): The context.
            target: The target to count.

        Returns:
            int: The count of the target following the context.
        """
        count = 0
        if context in self.observations and target in self.observations[context]:
            count = self.observations[context][target]
        return count

    def _get_probability(self, context, target):
        """
        Get the probability of a target given a context.

        Parameters:
            context (tuple): The context.
            target: The target.

        Returns:
            float: The probability of the target given the context.
        """
        total_count = self._count_context(context)
        if total_count == 0:
            return 0
        return self._count_context_target(context, target) / total_count

    def _get_likelihood(self, context):
        """
        Get the likelihood of all targets given a context.

        Parameters:
            context (tuple): The context.

        Returns:
            dict: A dictionary of targets and their probabilities.
        """
        return {target: self._get_probability(context, target) for target in self.alphabet}

    def _sample(self, context):
        """
        Sample a target note from the likelihood dictionary using weighted probabilities.

        Parameters:
            context (tuple): The context to sample from.

        Returns:
            The sampled target note.
        """
        likelihood = self._get_likelihood(context)
        targets, probabilities = zip(*likelihood.items())

        # Case where all probabilities are zero (No transition)
        if sum(probabilities) == 0:
            print(f"Warning: All probabilities are zero for context {context}.")
            return self.order_zero_model._sample(tuple())

        return random.choices(targets, probabilities)[0]

    def generate(self, size):
        """
        Generate a sequence of notes using the Markov Chain.

        Parameters:
            size (int): The length of the sequence to generate.

        Returns:
            list: The generated sequence of notes.
        """
        melody = []
        for _ in range(self.order): # Initial sampling, use zeroth order model
            melody.append(self.order_zero_model._sample(tuple()))
        for _ in range(size - self.order):
            context = tuple(melody[-self.order:])
            next_note = self._sample(context)
            melody.append(next_note)
        return melody

def save_melody_to_midi(melody, filename):
    """
    Save a melody to a MIDI file using music21.

    Parameters:
        melody (list of tuples): A list of (note, duration) pairs, where
                                 `note` is a MIDI pitch (or -1 for silence),
                                 and `duration` is the duration in quarter lengths.
        filename (str): The name of the MIDI file to save to.
    """
    midi_stream = stream.Stream()
    for pitch, duration in melody:
        if pitch == SILENCE:
            midi_stream.append(note.Rest(quarterLength=duration))
        else:
            midi_stream.append(note.Note(pitch, quarterLength=duration))
    midi_stream.write('midi', fp=filename)
    midi_stream.show()

if __name__ == '__main__':
    data = preprocessing()
    order = 2
    model = MarkovChain(order)
    model.train(data)
    melody = model.generate(40)
    save_melody_to_midi(melody, 'melody.mid')
