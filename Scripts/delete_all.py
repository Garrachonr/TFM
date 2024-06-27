from pymongo import MongoClient

# Establece la conexión con MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['memory_db']
memorable_data_collection = db['memorable_data']

resultado = memorable_data_collection.delete_many({})
print(f'Documentos eliminados: {resultado.deleted_count}')