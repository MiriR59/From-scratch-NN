import numpy as np
class tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        
    def encoder(self, list):
        id_list = []
        for word in list:
            id = []
            for i in range(len(word)):
                letter = word[i]
                for j, character in enumerate(self.vocabulary):
                    if letter == character:
                        id.append(j)
            id_list.append(id)
        return np.array(id_list)

    def decoder(self, ID):
        result = []

        for i, word in enumerate(ID):
            test = ''
            for token in word:
                character = self.vocabulary[token]
                test += character
            result.append(test)
        return result
