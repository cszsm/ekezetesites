from collections import deque
from sklearn.feature_extraction import DictVectorizer
import normalizer

vowel_table = {"a": ["á"], "e": ["é"], "i": ["í"], "o": ["ó", "ö", "ő"], "u": ["ú", "ü", "ű"]}


def deaccentize(text):
    text = text.replace("á", "a")
    text = text.replace("é", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o")
    text = text.replace("ö", "o")
    text = text.replace("ő", "o")
    text = text.replace("ú", "u")
    text = text.replace("ü", "u")
    text = text.replace("ű", "u")

    return text

def create_row(window, window_size):
    row = {}

    for i in range(-window_size, window_size + 1):
        row[i] = normalizer.normalize_character(deaccentize(window.popleft()))

    del row[0]

    return row


def prepare_text(text, window_size, vowel):
    x_e = []
    y_e = []
    lower_text = text.lower()

    window = deque((), window_size * 2 + 1)
    for i in range(window.maxlen):
        window.append("_")
        lower_text += "_"

    for character in lower_text:
        window.append(character)
        if window[window_size] == vowel:
            x_e.append(create_row(window.copy(), window_size))
            y_e.append([1, 0])
        if window[window_size] in vowel_table[vowel]:
            x_e.append(create_row(window.copy(), window_size))
            y_e.append([0, 1])

    # print(x_e)
    # print(y_e)
    return x_e, y_e

