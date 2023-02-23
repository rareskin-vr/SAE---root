import re, math
from collections import Counter
import numpy as np
import fuzzywuzzy.fuzz

WORD = re.compile(r'\w+')


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = np.dot(list(vec1.values()), list(vec2.values()))

    sum1 = np.sum(np.square(list(vec1.values())))
    sum2 = np.sum(np.square(list(vec2.values())))
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def givKeywordsValue(text1, text2):
    similarity_ratio = fuzzywuzzy.fuzz.token_sort_ratio(text1, text2)
    kval = 6 - math.floor(similarity_ratio / 20)
    return kval
