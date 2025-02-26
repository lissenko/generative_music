import os
import json
import re
from collections import defaultdict
import numpy as np
import tensorflow.keras as keras

DATASET_PATH = "rock_corpus_v1-1/"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "chords_file_dataset"
MAPPING_PATH = "chords_mapping.json"
SEQUENCE_LENGTH = 64

def load_songs_as_strings(dataset_path):
    """
    Loads all songs as strings from a given dataset directory.

    Args:
        dataset_path (str): Path to the dataset directory containing song files.

    Returns:
        list of str: List of song contents as strings.
    """
    songs = []
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "txt":
                with open(os.path.join(path, file), 'r') as file:
                    file_content = file.read()
                    songs.append(file_content)
    return songs

def trim_songs(songs):
    """
    Cleans a list of songs by parsing and removing unwanted sections.

    Args:
        songs (list of str): List of raw song strings.

    Returns:
        list of list: List of parsed and cleaned songs as tokenized sequences.
    """
    trimmed_songs = []
    for song in songs:
        trimmed_songs.append(parse_song(song))
    return trimmed_songs

def parse_song(data):
    """
    Parses a song into sections and expands its structure based on references.

    Args:
        data (str): Raw song text.

    Returns:
        list of str: Fully expanded sequence of chords.
    """
    sections = dict()
    lines = data.split('\n')
    
    # Filter out comment lines and blank lines
    cleaned_lines = [line.split('%')[0].strip() for line in lines if line.strip() and not line.strip().startswith('%')]
    cleaned_lines = [re.sub(r'\[.*?\]', '', line).strip() for line in cleaned_lines]

    for line in cleaned_lines:
        section = line.split(':')
        sections[section[0]] = section[1].split()

    def expand_section(section_name):
        """
        Recursively expands the tokens in a section.

        Args:
            section_name (str): Name of the section to expand.

        Returns:
            list of str: Expanded tokens from the section.
        """
        expanded_section = []
        for token in sections[section_name]:
            if token.startswith('$'):
                # Split the token to handle $reference*factor
                splited_token = token.split('*')
                referent_section = splited_token[0][1:]  # Remove the '$' character
                factor = int(splited_token[1]) if len(splited_token) == 2 else 1
                # Recursively expand the referenced section and repeat it by the factor
                expanded_section.extend(expand_section(referent_section) * factor)
            elif '*' in token:  # Handle tokens like |*n
                base_token, factor = token.split('*')
                factor = int(factor)
                expanded_section.extend([base_token] * factor)
            else:
                # Add tokens directly
                expanded_section.append(token)
        return expanded_section

    # Expand the 'S' section, resolving all references
    expanded_S = expand_section('S')
    return expanded_S

def write_progressions(progressions, filename, sequence_length=SEQUENCE_LENGTH):
    """
    Writes progressions to a single file, separating songs with a delimiter.

    Args:
        progressions (list of list of str): List of tokenized chord sequences.
        filename (str): Path to the file where progressions will be saved.
        sequence_length (int, optional): Length of sequences to determine the delimiter. Default is SEQUENCE_LENGTH.
    """
    new_song_delimiter = "/ " * sequence_length
    with open(filename, 'w') as file:
        for i, progression in enumerate(progressions):
            delimiter = new_song_delimiter if i != len(progressions) else new_song_delimiter[:-1]
            file.write(' '.join(progression) + ' ' + delimiter)

def create_mapping(single_file_dataset, mapping_path):
    """
    Creates a mapping from chord symbols to integers and saves it to a JSON file.

    Args:
        single_file_dataset (str): Path to the dataset file containing all chords.
        mapping_path (str): Path to the JSON file where the mappings will be saved.
    """
    mappings = {}
    with open(single_file_dataset, 'r') as file:
        songs = file.read().split()
    alphabet = set(songs)
    for i, symbol in enumerate(alphabet):
        mappings[symbol] = i
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

def preprocess(dataset_path):
    """
    Preprocesses the dataset by loading and trimming all songs.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        list of list of str: List of parsed and tokenized songs.
    """
    songs = load_songs_as_strings(dataset_path)
    songs = trim_songs(songs)
    return songs

def load(file_path):
    """
    Loads the content of a file as a string.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Content of the file as a string.
    """
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def convert_songs_to_int(songs):
    """
    Converts songs (chord sequences) to integer sequences using a mapping.

    Args:
        songs (str): All songs in a single string format.

    Returns:
        list of int: List of integer-encoded chord sequences.
    """
    int_songs = []

    # Load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # Transform songs string to list
    songs = songs.split()

    # Map songs to integers
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def create_data_generator(sequence_length, batch_size):
    """
    Creates a data generator for training the LSTM model.

    Args:
        sequence_length (int): Length of input sequences.
        batch_size (int): Number of samples per batch.

    Yields:
        tuple: Batch of input sequences (numpy array) and corresponding targets (numpy array).
    """
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    while True:
        # Randomly sample batches
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            # Randomly select a starting point
            start = np.random.randint(0, len(int_songs) - sequence_length - 1)
            # Create input sequence and target
            input_seq = int_songs[start:start+sequence_length]
            target = int_songs[start+sequence_length]
            batch_inputs.append(input_seq)
            batch_targets.append(target)
        yield np.array(batch_inputs), np.array(batch_targets)

def main():
    """
    Main function that preprocesses the dataset, writes all songs to a single file, and creates chord mappings.
    """
    songs = preprocess(DATASET_PATH)
    write_progressions(songs, SINGLE_FILE_DATASET)
    create_mapping(SINGLE_FILE_DATASET, MAPPING_PATH)

if __name__ == '__main__':
    main()

