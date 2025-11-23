import numpy as np
import nltk
import string

def tokenize(sentence):
    """
    делим предложение на массив слов 
    """
    tokens = nltk.word_tokenize(sentence, language='russian')
    # оставляем только слова и числа
    return [word for word in tokens if word not in string.punctuation + '?!.,']

def stem(word):
    """
    для русского языка лучше использовать просто нижний регистр
    """
    return word.lower()

def bag_of_words(tokenized_sentence, words):

    # нижний регистр
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

# тест
if __name__ == "__main__":
    test_text = "Привет! Как дела? С Новым 2024 годом!"
    tokens = tokenize(test_text)
    print("Токены:", tokens)
    
    stems = [stem(word) for word in tokens]
    print("Нормализованные:", stems)