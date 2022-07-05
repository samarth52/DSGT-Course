from numpy import argmax
sentence = "i am a rambling wreck from georgia tech"
alphabet = 'abcdefghijklmnopqrstuvwxyz '
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
integer_encoded = []
for char in sentence:
    integer_encoded.append(char_to_int[char])
print(integer_encoded)
onehot_encoded = []
letter = [0 for _ in range(len(alphabet))]
for value in integer_encoded:
    letter[value] = letter[value]+1
onehot_encoded.append(letter)
print(onehot_encoded)

print(argmax(onehot_encoded[0]))
print(int_to_char)
inverted = int_to_char[argmax(onehot_encoded[0])]
print(inverted)