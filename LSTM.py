from pickle import dump, load
import pandas
import sqlite3

database = r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\Articles\articles.db'
df = pandas.read_csv(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\Articles\news_summary_more.csv')


class SQL:
    def __init__(self, database):
        self.database = database
        self.conn = sqlite3.connect(database)
  
    def create_database(self, db_name):
        self.conn.execute('drop table if exists ' + db_name + ' ;')
        self.conn.execute('CREATE TABLE ' + db_name + ' (headlines TEXT, text TEXT)')
        
    def import_data(self, db_name,  df):   
        df.to_sql(db_name, self.conn, if_exists='append', index=False)
    
    def get_sample(self, sample_size):
        import random
        import pandas as pd
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM articles;")
         
        rows = cur.fetchall()
        length = len(rows)
        random_init = random.randint(sample_size, length)
    
        if random_init+sample_size > length:
            sample = rows[random_init-sample_size:random_init]
        else:    
            sample = rows[random_init:random_init+sample_size]
        
        df_articles = pd.DataFrame(sample, columns = ['summary', 'text'])
        
        return df_articles
 
sql = SQL(database)    
sql.create_database('buba')
sql.import_data('buba', df)
x = sql.get_sample(1000)



####################################################################
contractions = {
 "ain't": "am not",
 "aren't": "are not",
 "can't": "cannot",
 "can't've": "cannot have",
 "'cause": "because",
 "could've": "could have",
 "couldn't": "could not",
 "couldn't've": "could not have",
 "didn't": "did not",
 "doesn't": "does not",
 "don't": "do not",
 "hadn't": "had not",
 "hadn't've": "had not have",
 "hasn't": "has not",
 "haven't": "have not",
 "he'd": "he would",
 "he'd've": "he would have"}

def clean_text(text, remove_stopwords=True):
     # Convert words to lower case
     text = text.lower()
     
     if True:
         text = text.split()
         new_text = []
         for word in text:
             if word in contractions:
                 new_text.append(contractions[word])
             else:
                 new_text.append(word)
         text = " ".join(new_text)
     
     text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
     text = re.sub(r'\<a href', ' ', text)
     text = re.sub(r'&amp;', '', text)
     text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
     text = re.sub(r'<br />', ' ', text)
     text = re.sub(r'\'', ' ', text)
     
     if remove_stopwords:
         text = text.split()
         stops = set(stopwords.words("english"))
         text = [w for w in text if not w in stops]
         text = " ".join(text)
     
     return text
 
import re 
stories = list()
for index, row in x.iterrows():
    row['text'] =  clean_text(str(row['text']), remove_stopwords=True)
    row['summary'] =  clean_text(str(row['summary']), remove_stopwords=True)
    stories.append({'story': row['text'], 'highlights': row['summary']})

# save to file
dump(stories, open(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\Articles\article_dataset.pkl', 'wb'))


batch_size = 64
epochs = 110  
latent_dim = 256  
num_samples = 10000 

stories = load(open(r'C:\Users\Konrad\Desktop\NOVA IMS\text mining\Articles\article_dataset.pkl', 'rb'))
print('Loaded Stories %d' % len(stories))
print(type(stories))


input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
for story in stories:
     input_text = story['story']
     target_text =story['highlights']
     
     # We use "tab" as the "start sequence" character
     # for the targets, and "\n" as "end sequence" character.
     target_text = '\t' + target_text + '\n'
     input_texts.append(input_text)
     target_texts.append(target_text)
     for char in input_text:
         if char not in input_characters:
             input_characters.add(char)
     for char in target_text:
         if char not in target_characters:
             target_characters.add(char)
             
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Model

def define_models(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,  initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model



# define model
train, infenc, infdec = define_models(345, 63, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# generate training dataset
X1 = np.array(input_texts)
y = np.array(target_texts)
 y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# train model
train.fit(X1, y, epochs=1)
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))

