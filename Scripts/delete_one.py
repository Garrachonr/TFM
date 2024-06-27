from pymongo import MongoClient
from datetime import datetime

# Establece la conexión con MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['memory_db']
memorable_data_collection = db['memorable_data']

dato = "User likes video games"
resultado = memorable_data_collection.delete_one({'data': dato})
print(f'Documento eliminado: {resultado.deleted_count}')