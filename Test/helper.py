from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer()


# generates template windows for the alphabet
def generate_windows(window_size):
    windows = []
    alphabet = "abcdefghijklmnopqrstuvwxyz 0_*"
    alphabet_size = len(alphabet)

    for i in range(alphabet_size):
        new_window = {}

        end_of_slice = i + window_size * 2
        if end_of_slice <= alphabet_size:
            alphabet_slice = alphabet[i:end_of_slice]
        else:
            alphabet_slice = alphabet[i:alphabet_size]
            alphabet_slice += alphabet[0:end_of_slice - alphabet_size]

        for j in range(window_size):
            new_window[-1 * (j + 1)] = alphabet_slice[window_size - 1 - j]
            new_window[j + 1] = alphabet_slice[window_size + j]

        windows.append(new_window)

    return windows
