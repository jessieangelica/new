# ner_model.py
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from transformers import DistilBertTokenizer
import pandas as pd

def compute_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return loss_fn(tf.reshape(labels, [-1]), tf.reshape(logits, [-1, 2]))

def load_ner_model(model_path, tokenizer_path):
    model = load_model(model_path, custom_objects={'compute_loss': compute_loss})
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def load_dictionary(csv_path):
    # Load and process the dictionary from CSV
    keywords_df = pd.read_csv(csv_path)
    temp = dict(zip(keywords_df['category'],keywords_df['label']))
    keyword = {key.lower(): value.upper() for key, value in temp.items()}
    splitted_keyword = [key.split() for key in keyword.keys()]
    return keyword, splitted_keyword

def setup_keyvalue(word, keyword, splitted_keyword):
    # Setup key-value function inferencer
    if word in keyword:
      return keyword[word]
    if len(word) > 1 and word[0] == '$' or word[0] == '$':
      return 'MONEY'
    for i in splitted_keyword:
      if word in i:
        return keyword[' '.join(i)];
    return 'O'

def scan_punctuation(money):
    # Money detector function
    i = len(money)-1
    if money[i] == '.' or money[i] == ',':
      return money[:-1]
    return money

def scan_money(text):
    # Scan money in the text
    temp = text.split()
    for i in range(len(temp)):
      if temp[i][0] == '$':
        if len(temp[i]) == 1:
          if '1' <= temp[i+1][0] and temp[i+1][0] <= '9':
            return scan_punctuation(temp[i+1])
        else:
          if '1' <= temp[i][1] and temp[i][1] <= '9':
            return scan_punctuation(temp[i][1:len(temp[i])])
    return None

def make_prediction(model, tokenizer, text, key_value_function, keyword, splitted_keyword):
    # Prediction maker function
    encodings = tokenizer(text, truncation=True, padding=True)

    # Convert encoded inputs to tensors
    input_ids = tf.constant([encodings['input_ids']])  # Wrap the input_ids within a list to create a batch
    attention_mask = tf.constant([encodings['attention_mask']])  # Wrap the attention_mask within a list to create a batch

    # Make predictions
    predictions = model({'input_ids': input_ids, 'attention_mask': attention_mask}, training=False)

    # Extract logits from predictions
    logits = predictions['logits']

    # Process predictions
    money = int(scan_money(text))
    predicted_labels = tf.argmax(logits, axis=-1).numpy()
    tokens = tokenizer.convert_ids_to_tokens(encodings['input_ids'])  # Accessing the first sentence
    entities = []
    non_money_entity_label = None # Used to send back the entity label detected to the return message JSON
    current_entity = None
    for j, token in enumerate(tokens):
        if predicted_labels[0][j-1] == 1:  # Assuming 1 corresponds to the entity label of the first sentence in the batch
            if current_entity:
                current_entity += " " + token
            else:
                current_entity = token
                # non_money_entity_label = None # Reset the none money entity label
                if non_money_entity_label is None:
                  non_money_entity_label = key_value_function(current_entity, keyword, splitted_keyword)
        else:
          if current_entity:
            if len(current_entity.split()) > 1:
              temp = current_entity.split()
              for i in range(len(temp)):
                temp2 = key_value_function(temp[i], keyword, splitted_keyword)
                # temp2 = setup_keyvalue(temp[i])
                if temp2 != 'O' or i == len(temp)-1:
                  entities.append((current_entity, temp2))
                  current_entity = None
                  break
            else:
              # entities.append((current_entity, setup_keyvalue(current_entity)))
              entities.append((current_entity, key_value_function(current_entity, keyword, splitted_keyword)))
              current_entity = None
    if current_entity:
        entities.append((current_entity, non_money_entity_label))
    # return text, entities, money, non_money_entity_label # original DO NOT DELETE
    return entities, non_money_entity_label
