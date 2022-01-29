import pickle
import numpy as np

with open('train_qa.txt','rb') as f:
    train_data = pickle.load(f)
    
with open('test_qa.txt','rb') as f:
    test_data = pickle.load(f)

# setting up a vocabulary 
all_data = test_data + train_data
#in order to create the vocab we create a set which an UNORDERED COLLECTION OF UNIQUE DATA
vocab = set()

for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
    
vocab.add('no')
vocab.add('yes')

vocab_len = len(vocab) + 1
#this is done for keras pad sequences

# Figuring out the longest story and longest question again for keras pad sequences
all_story_lens = [len(data[0]) for data in all_data]

max_story_len = max(all_story_lens)
max_question_len = max([len(data[1]) for data in all_data])

#Vectorizing the data
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters = [])
tokenizer.fit_on_texts(vocab)

# Creating a function to tokenize the stories,questions and answers
def vectorize_stories(data, word_index = tokenizer.word_index, max_story_len = max_story_len, max_question_len = max_question_len):
    #Strories
    X = []
    #Questions
    Xq = []
    # Answers
    Y = []
    
    for story, query, answer in data:
        # for each story
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        
        y = np.zeros(len(word_index)+1)
        y[word_index[answer]] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
    return (pad_sequences(X, maxlen = max_story_len), pad_sequences(Xq, maxlen = max_question_len), np.array(Y))
        
inputs_train, questions_train, answers_train = vectorize_stories(train_data)
inputs_test, questions_test, answers_test = vectorize_stories(test_data)

# Build the neural network (input encoder M and C ,and the Question Encoder)
from keras.models import Sequential , Model 
from keras.layers.embeddings import Embedding
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM

#create place folders using input to bind story and question under the answer label
input_sequence = Input((max_story_len,))
question = Input((max_question_len,))

# Defining the vocabulary size
vocab_size = len(vocab) + 1

# Input Encoder M
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_size,output_dim = 64))
input_encoder_m.add(Dropout(0.4)) # This layer will drop 40% of the neurons while training in order to prevent overfitting of the model 
# The encoderoutput will be in the form of (samples, story_max_len, embedding_dim)

# Input Encoder C
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_size,output_dim = max_question_len))
input_encoder_c.add(Dropout(0.5)) # This layer will drop 50% of the neurons while training in order to prevent overfitting of the model 
# The encoderoutput will be in the form of (samples, story_max_len, question_max_len)

#Question encoder
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_size,output_dim = 64, input_length = max_question_len))
question_encoder.add(Dropout(0.4)) # This layer will drop 40% of the neurons while training in order to prevent overfitting of the model 
# The encoderoutput will be in the form of (samples, question_max_len, embedding_dim)

#ENCODED == Encoder(input)
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

#dot product and andding softmax activation
match = dot([input_encoded_m, question_encoded], axes=(2,2))
match = Activation('softmax')(match)

response = add([match, input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer) #(samples, vocab_size) #YES\NO 0000

answer = Activation('softmax')(answer)

model = Model((input_sequence, question), answer)

model.compile(optimizer = 'rmsprop',loss ='categorical_crossentropy', metrics = ['accuracy'])

model.summary()

# fit/train the network 
history = model.fit([inputs_train, questions_train], answers_train,batch_size=16,epochs=120,validation_data=([inputs_test, questions_test], answers_test))

#plotting the training history
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Prediction
pred_results = model.predict(([inputs_test, questions_test]))

# formatting the pred
# Note the whitespace of the periods
my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()
        
my_question = "Is the football in the garden ?"
my_question.split()

mydata = [(my_story.split(),my_question.split(),'yes')]

my_story,my_ques,my_ans = vectorize_stories(mydata)

pred_results = model.predict(([ my_story, my_ques]))

val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])
