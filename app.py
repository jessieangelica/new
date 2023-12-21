from fastapi import FastAPI, HTTPException, Form
# from db import SessionLocal, create_expense

import os 

from ner_model import load_ner_model, load_dictionary, setup_keyvalue, scan_money, make_prediction

from pydantic import BaseModel

from typing import Optional

app = FastAPI()

# Load our NER model and dictionary
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'models', 'Intellicash_Model_S2V5', 'content', 'Intellicash_Model_S2V5')
tokenizer_path = os.path.join(base_path, 'models', 'TOKENIZER', 'content', 'TOKENIZER')
keyword_path = os.path.join(base_path, 'models', 'Intellicash_Label.csv')

# Extract necessary components from the loaded dictionary
ner_model, ner_tokenizer = load_ner_model(model_path, tokenizer_path)
keyword, splitted_keyword = load_dictionary(keyword_path)

# Pass the components as arguments to setup_keyvalue
key_value_function = lambda word, keyword, splitted_keyword: setup_keyvalue(word, keyword, splitted_keyword)

class PredictionRequest(BaseModel):
    text: str

class PredictionResult(BaseModel):
    text: str
    entities: list
    money: Optional[int]
@app.get("/")
def read_root():
    return {"Hello": "Welcome to Intellicash"}

@app.post("/predict/")
async def predict(request: PredictionRequest):
    text = request.text
    money = scan_money(text)
    # entities = make_prediction(ner_model, ner_tokenizer, text, key_value_function, keyword, splitted_keyword)
    entities, non_money_entity_label = make_prediction(ner_model, ner_tokenizer, text, key_value_function, keyword, splitted_keyword)
    print(f"Received request with text: {text}")

    if money is not None and entities and non_money_entity_label is not None:
        # db = SessionLocal()
        # create_expense(db, text, entities, money)
        # db.close()
        return {
            'status': {
                'code': 200,
                # 'message': 'You have succesfully added $amount to $category',
                # 'message': f'You have successfully added ${money} to {entities}',
                'message': f'You have successfully added ${money} to {non_money_entity_label}',
                'result': PredictionResult(text=text, entities=entities, money=money).dict()
            }
        }
    else:
        raise HTTPException(
            status_code=300,
            detail='Either Category or Cost is not detected'
        )