import pickle

DATA_FILE = 'jsb-chorales-quarter.pkl'
SOPRANO_VOICE_IDX = -1
SILENCE = -1

def get_data_split(split, data_file=DATA_FILE):
    """
    Load a specific data split (train, test, or validation) from the dataset.

    Parameters:
        split (str): The data split to load ('train', 'test', or 'valid').
        data_file (str): The path to the dataset file.

    Returns:
        list: A list of musical pieces, each represented as a list of notes.
    """
    with open(data_file, 'rb') as p:
        data = pickle.load(p, encoding="latin1")
        data_split = data[split]
    return data_split

def get_soprano_voice(data):
    """
    Extract the soprano voice (the highest-pitched notes) from the dataset.

    Parameters:
        data (list): A list of musical pieces, each represented as a list of notes.

    Returns:
        list: A list of soprano voices, where each voice is a list of notes or silence (-1).
    """
    return [[note[SOPRANO_VOICE_IDX] if len(note) != 0 else SILENCE for note in piece] for piece in data]

def get_notes_and_dur(data):
    """
    Convert a sequence of notes into note-duration pairs.

    Parameters:
        data (list): A list of musical pieces, each represented as a list of notes.

    Returns:
        list: A list of musical pieces, where each piece is a list of (note, duration) pairs.
    """
    return_data = []
    for piece in data:
        note_dur_piece = []
        i = 0
        while i < len(piece): # Check wether the note spans on more than a quarter note
            note = piece[i]
            duration = 1
            j = i + 1
            while j < len(piece) and piece[j] == note:
                j += 1
                i += 1
                duration += 1
            i += 1
            note_dur_piece.append((note, duration))
        return_data.append(note_dur_piece)
    return return_data

def preprocessing():
    """
    Preprocess the training data by extracting the soprano voice and converting
    it into note-duration pairs.

    Returns:
        list: Preprocessed training data.
    """
    data = get_data_split('train')
    data = get_soprano_voice(data)
    data = get_notes_and_dur(data)
    return data

