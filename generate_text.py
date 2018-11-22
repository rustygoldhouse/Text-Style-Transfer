
'''


>>> python3 generate_text.py

to generate text in the style of training data.


'''

def generate_test():
    import time
    sentence='Our first principle was: pack and contagion, the contagion of the pack, such is the path becoming-animal takes. But a second principle seemed to tell us the opposite: wherever there is multipli- city, you will also find an exceptional individual, and it is with that individ- ual that an alliance must be made in order to become-animal. There may be no such thing as a lone wolf, but there is a leader of the pack, a master of the pack, or else the old deposed head of the pack now living alone, there is the Loner, and there is the Demon.'
    sentence=sentence.split()
    sentence=[words_to_code[w] for w in sentence]
    sentence=np.array(sentence)
    sentence=sentence.reshape(1,5)
    
    model=load_model('newtextmodel.h5')
    
    while True:
        x=np.argmax(model.predict(sentence))
        print(code_to_words[x],end=' ')
        sentence=sentence.flatten()
        sentence=sentence.tolist()
        sentence.append(x)
        sentence=sentence[1:]
        sentence=np.array(sentence)
        sentence=sentence.reshape(1,5)
        time.sleep(1)




"""
from keras.models import load_model
import time
import numpy as np

file=open(u'kafka.txt', 'r', encoding='utf-8')
string=file.read()
words=string.split()

sentences=[words[i:i+5] for i in range(25000)]
targets=[words[i+5] for i in range(25000)]


vocab=list(set(words))

words_to_code=dict((i,j) for j,i in enumerate(vocab))
code_to_words=dict((i,j) for i,j in enumerate(vocab))

def generate_test():
    sentence=str(u'Our first principle was: pack and contagion, the contagion of the pack, such is the path becoming-animal takes. But a second principle seemed to tell us the opposite: wherever there is multipli- city, you will also find an exceptional individual, and it is with that individ- ual that an alliance must be made in order to become-animal. There may be no such thing as a lone wolf, but there is a leader of the pack, a master of the pack, or else the old deposed head of the pack now living alone, there is the Loner, and there is the Demon.')
    sentence=sentence.split()
    sentence=[words_to_code[w] for w in sentence]
    sentence=np.array(sentence)
    sentence=sentence.reshape(1,5)
    
    model=load_model('newtextmodel.h5')
    
    while True:
        x=np.argmax(model.predict(sentence))
        print(code_to_words[x],end=' ')
        sentence=sentence.flatten()
        sentence=sentence.tolist()
        sentence.append(x)
        sentence=sentence[1:]
        sentence=np.array(sentence)
        sentence=sentence.reshape(1,5)
        time.sleep(1)
"""
        
if __name__ == '__main__':
    generate_test()
