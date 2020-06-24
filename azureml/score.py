import os
import pickle
import json
import numpy as np
import tensorflow as tf

def split_text(text):
  return text.replace(',',' ').replace(';',' ').replace('.',' ').replace('!',' ').split()

def corpus_to_ids(corpus, max_words_per_sent=500):
  lines = []
  for text in corpus:
      words = split_text(text)
      ids = [word_to_id.get(word, UNK_WORD_ID) for word in words]
      ids = ids[:max_words_per_sent]
      if len(ids) < max_words_per_sent:
          ids += [0] * (max_words_per_sent - len(ids))
      lines.append(ids)
  return np.array(lines)

#Called when the deployed service starts
def init():
  global model
  global vocab
  global counts
  global UNK_WORD_ID
  global word_to_id
  global id_to_word
  
  #Get the path to the model
  model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), './models')
  
  #load models
  model = tf.keras.models.load_model(os.path.join(model_path, 'restocracy.h5'))
  
  #load vocab + counts
  with open(os.path.join(model_path, 'vocab.pkl'), 'rb') as handle:
    vocab = pickle.load(handle)
    
  with open(os.path.join(model_path, 'counts.pkl'), 'rb') as handle:
    counts = pickle.load(handle)
  
  word_to_id = vocab
  id_to_word = {v:k for k,v in word_to_id.items()}
  UNK_WORD_ID = word_to_id['UNK']
  return

#Handle requests to the service
def run(data):
  try:
    print('Received data: {}'.format(data))
    #this method expects a call of the form {"text": "am mancat la un restaurant super fancy"}
    data = json.loads(data)
    assert isinstance(data, dict)
    text = data['text']
    np_data = corpus_to_ids([text])
    prediction = model.predict(np_data)
    return json.dumps(prediction.tolist())
  except Exception as e:
    return str(e)

  
if __name__ == '__main__':
  os.environ['AZUREML_MODEL_DIR'] = ''
  init()
  payload = json.dumps({'text': 'am mancat super biner'})
  preds = run(payload)
  print(preds)
  
  
  
  