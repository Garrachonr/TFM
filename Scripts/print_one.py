from pymongo import MongoClient

# Establece la conexión con MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['memory_db']
memorable_data_collection = db['memorable_data']

# Función para obtener un documento específico
def obtener_documento(data):
    documento = memorable_data_collection.find_one({'data': data})
    if documento:
        print("Documento encontrado:", documento)
    else:
        print("No se encontró ningún documento con esos datos.")

# Ejemplo de cómo obtener un documento
data = "User likes video games"
obtener_documento(data)
