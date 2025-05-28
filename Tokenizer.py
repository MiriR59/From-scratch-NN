class tokenizer:
    def __init__(self):
        self.vocabulary = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
        
    def encoder(self, sentence):
        id = []
        
        for i in range(len(sentence)):
            letter = sentence[i]
            for j, character in enumerate(self.vocabulary):
                if letter == character:
                    id.append(j)
        return id

    def decoder(self, ID):
        test = ''
        
        for token in ID:
            character = self.vocabulary[token]
            test += character
        return test
