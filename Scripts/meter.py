from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime


#Model and tokenizer
model_name = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Conexión a MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['memory_db']
    memorable_data_collection = db['memorable_data']
except Exception as e:
    print(f"Error al conectar con MongoDB: {e}")
    exit(1)

def get_embedding(text):
    """ Genera un embedding del texto dado utilizando el modelo de Hugging Face """
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    #Promediar las salidas del último layer para obtener un único vector de embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.flatten().tolist()

memorable_items = [
    "User play tennis since he was 10 years old",
    "Has a pet dog named Max",
    "User lived for 5 years in Japan",
    "Max is a golden retriever",
    "User injured his knee while playing tennis",
    "User likes video games",
    "Likes pokemon and zelda, big fan of nintendo",
    "Has a ps4 and a switch",
    "went to the azores on vacation",
    "likes to read science fiction books",
    ]

memorable_data = [{'data': item, 'embedding': get_embedding(item.strip()), 'timestamp': datetime.now().isoformat(), 'count': 1} for item in memorable_items]

for new_data in memorable_data:
    memorable_data_collection.insert_one(new_data)

print("Memorable data inserted successfully")