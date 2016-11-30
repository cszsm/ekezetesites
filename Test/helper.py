from collections import deque

vowel_table = {"a": ["á"], "e": ["é"], "i": ["í"], "o": ["ó", "ö", "ő"], "u": ["ú", "ü", "ű"]}


def create_row(window, window_size):
    row = {}

    for i in range(-window_size, window_size + 1):
        row[i] = window.popleft()

    del row[0]

    return row


def prepare_text(text, window_size, vowel):
    x_e = []
    y_e = []

    window = deque((), window_size * 2 + 1)
    for i in range(window.maxlen):
        window.append("_")

    for character in text:
        window.append(character)
        if window[window_size] == vowel:
            x_e.append(create_row(window.copy(), window_size))
            y_e.append(0)
        if window[window_size] in vowel_table[vowel]:
            x_e.append(create_row(window.copy(), window_size))
            y_e.append(1)

    # print(x_e)
    # print(y_e)
    return x_e
