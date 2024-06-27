from pymongo import MongoClient

# Establece la conexión con MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['memory_db']
memorable_data_collection = db['memorable_data']

documentos = memorable_data_collection.find({})
for documento in documentos:
    print(documento)