import pandas as pd
import numpy as np
import email
import re

def extract_plaintext_content(raw_email):
    email_object = email.message_from_string(raw_email)
    plain_text_parts = []

    for part in email_object.walk():
        if part.get_content_type() == 'text/plain':
            plain_text_parts.append(part.get_payload())
    return ''.join(plain_text_parts)


def containNumbers(text):
    return bool(re.search(r'\d', text))


def notContainSpecialCharacters(text):
    regexp = re.compile("""[-+=*@_#`"$%^&*[]()<>/|}{~:]""")
    if (regexp.search(text) == None):
        return True
    else:
        return False

def removeShortForms(text):
    """
    Returns decontracted phrases
    """
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

def removeSpacesAndConvertToLowercase(text):
    # removes anything other than alphabets
    text = re.sub('[^A-Za-z]+', ' ', text)

    # Remove extra spaces in between, leading and trailaing spaces and covert to lowercase
    return ' '.join(text.split()).strip().lower()

def getPreprocessedSentencesFromEmail(email):
    sentences = []

    for sentence in email.split('.'):
        sentence = sentence.strip()
        no_of_words = len(sentence.split())

        if 3 <= no_of_words <= 25 \
            and sentence[0].isupper() and sentence[1:].islower() \
            and not containNumbers(sentence) \
            and notContainSpecialCharacters(sentence):
                sentence = removeShortForms(sentence)

                for j in re.split('(?<=[?]) +', sentence):
                    sentences.append(removeSpacesAndConvertToLowercase(j))

    return pd.DataFrame({'sentence':sentences})


def applyPreprocessingStepsOnAllEmails(emails):
    result_df = pd.DataFrame()
    for index in range(emails.shape[0]):
        print(index)
        if index == 0:
            result_df = getPreprocessedSentencesFromEmail(emails.content.iloc[0])
        else:
            result_df = pd.concat([result_df,
            getPreprocessedSentencesFromEmail(emails.content.iloc[index])],
            ignore_index=True)

    result_df = result_df.drop_duplicates()
    return result_df

# Part 1: preprocess data
'''
emails_raw = pd.read_csv('emails.csv')
print(emails_raw.shape)
emails = pd.DataFrame()
# Потренироваться на переборах такого типа
emails['content'] = [extract_plaintext_content(i) for i in emails_raw['message']]
#print(emails_raw.head())
#print(emails_raw.shape)
#print(emails_raw['message'].iloc[1])
#sample_email = emails_raw['message'].iloc[1]
sentences = applyPreprocessingStepsOnAllEmails(emails)
sentences.to_csv('sentences.csv', index=False, index_label=True)
'''

# Part 2: train neuronet
sentence_df = pd.read_csv('sentences.csv')
sentence_df = sentence_df.dropna()
sentence_df.head()

#for sentence in sentence_df.sentence.sample(10):
#    print(sentence)

sentences = sentence_df.sentence.values
print("Total number of sentence: ", len(sentences))
print(sentences[0:10])

'''
from tensorflow.keras.preprocessing.text import Tokenizer
test_tokenizer = Tokenizer()
test_sentences = ['here is our forecast',
                  'especially if you have to prepare a presentation']
test_tokenizer.fit_on_texts(test_sentences)
print(test_tokenizer.word_index)
print(test_tokenizer.index_word)


test_sentence = "here you have our presentation"
test_token_list = test_tokenizer.texts_to_sequences([test_sentence])[0]
print(test_token_list)

n_grams = []
for i in range(1, len(test_token_list)):
    n_gram = test_token_list[:i+1]
    n_grams.append(n_gram)
print(n_grams)
'''

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
# Text sequences and n-grams
def convertSentencesIntoSeqOfTokens(sentences):
    tokenizer.fit_on_texts(sentences)
    total_words_in_vocab = len(tokenizer.word_index) + 1 # Why +1?

    input_sequences = []
    for sentence in sentences:
        # Why 0?
        seq_of_tokens = tokenizer.texts_to_sequences([sentence])[0]
        for i in range(1, len(seq_of_tokens)):
            n_gram = seq_of_tokens[:i+1]
            input_sequences.append(n_gram)

    return input_sequences, total_words_in_vocab

input_sequences, total_words_in_vocab = convertSentencesIntoSeqOfTokens(sentences)
print(input_sequences[:10])

# Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
test_sequences = [[2025, 2], [2025, 2, 16], [2025, 2, 16, 6],
                  [2025, 2, 16, 6, 135], [2025, 2, 16, 6, 135, 119]]
pad_sequences(test_sequences, maxlen=6, padding='pre')

def generateSameLengthSentencesByPadding(sequences):
    # Find len of longest sequence
    max_seq_len = max([len(x) for x in sequences])

    # Pad
    padded_sequences = np.array(pad_sequences(sequences, maxlen=max_seq_len, padding='pre'))

    return padded_sequences, max_seq_len

padded_sequences, max_seq_len = generateSameLengthSentencesByPadding(input_sequences)
#print('Padding:')
#print(padded_sequences[:5])

# Generate predictors and labels for training
import tensorflow.keras.utils as ku


#test_padded_sequences = np.array([[0, 0, 0, 12, 16, 32],
#                                  [0, 0, 8, 15, 17, 41]])
# print(test_padded_sequences[:,:-1])

def generatePredictorsAndLabels(padded_sequences):
    inputs, label = padded_sequences[:,:-1], padded_sequences[:,-1]
    label = ku.to_categorical(label, num_classes = total_words_in_vocab)
    return inputs, label

inputs, label = generatePredictorsAndLabels(padded_sequences)

input_length = max_seq_len - 1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
model = Sequential()
model.add(Embedding(total_words_in_vocab, 10, input_length=input_length))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(total_words_in_vocab, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())

model.fit(inputs, label, epochs=100)
model.save('lstm_text_autocomplete')

from tensorflow.keras.models import load_model
model = load_model('lstm_text_autocomplete')

def generate_autocomplete_suggestions(seed_sentence, no_of_next_words,
                                      model, max_sequence_len):
    for _ in range(no_of_next_words):
        sequence = tokenizer.texts_to_sequences([seed_sentence])[0]

        padded_sequence = pad_sequences([sequence],
                                        maxlen=max_seq_len-1,
                                        padding='pre')

        predictions = model.predict(padded_sequence, verbose=0)

        predicted_label = np.argmax(predictions, axis=1)[0]

        next_word = tokenizer.index_word[predicted_label]

        seed_sentence += " "+ next_word

    return seed_sentence

# Check results
print (generate_autocomplete_suggestions("In response to your earlier email", 10,
                                         model, max_seq_len))

print (generate_autocomplete_suggestions("I am happy to", 10,
                                         model, max_seq_len))

print (generate_autocomplete_suggestions("What is the status", 3,
                                         model, max_seq_len))

print (generate_autocomplete_suggestions("Here is the data", 3,
                                         model, max_seq_len))

print (generate_autocomplete_suggestions("Thank you very much", 4,
                                         model, max_seq_len))

print (generate_autocomplete_suggestions("I got your email", 17,
                                         model, max_seq_len))
