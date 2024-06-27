import requests
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import ollama
import torch
import openai
import argparse
import re


#Global variable
log = ""

######################
###### MongoDB #######
######################

#Connecting to MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['memory_db']
    memorable_data_collection = db['memorable_data']
except Exception as e:
    print(f"Error when connecting to MongoDB: {e}")
    exit(1)

#Retrieve memorable data from MongoDB
def get_memorable_data():
    try:
        return list(memorable_data_collection.find())
    except Exception as e:
        print(f"Error when getting data from MongoDB: {e}")
        return []
    

###############################
###### Model Embeddings #######
###############################

#Model and tokenizer for generating embeddings
model_name = 'mixedbread-ai/mxbai-embed-large-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

####################################
###### Extract Memories LLM ########
####################################

#API call for memorizing
def memory_api(text, memorize_type):
    try:

        #Adjust prompt depending if only memorizing the turns of the user or all the turns of the conversation
        if memorize_type == "user":

            prompt = """Given the following turns of a conversation with a user, identify and list memorable personal facts and data in short phrases as a human inside that conversation would.
        Reflect on what are the data or concepts from the conversation about the user that are valuable to memorize.
        Pay attention to the data related to personal experience and atributes from the user.
        You must list only all the memorable information where each memorable information is enclosed by "/p" at the beginning and "/p" at the end. Return nothing else.
        You must pay attention and follow the structure, return each memorable information enclosed with "/p" at the beginning and "/p" at the end of that memorable information. Following this structure is very important and a must.
        Here are the turns of the user:"""
            
        else:

            prompt = """Given the following conversation with a user, identify and list memorable personal facts and data in short phrases as a human inside that conversation would.
        Reflect on what are the data or concepts from the conversation about the user that are valuable to memorize.
        Pay attention to the data related to personal experience and atributes from the user.
        You must list only all the memorable information where each memorable information is enclosed by "/p" at the beginning and "/p" at the end. Return nothing else.
        You must pay attention and follow the structure, return each memorable information enclosed with "/p" at the beginning and "/p" at the end of that memorable information. Following this structure is very important and a must.
        Here is the conversation:"""

        full_prompt = f"Prompt: {prompt}. Conversation: {text}"
        response = ollama.generate(model="llama2:13b", prompt=full_prompt)
        return response
    except Exception as e:
        print(f"Error when extracting memorable data with Ollama: {e}")

#ReGeX for extracting data from the response
def extract_data(text):

    #Pattern for extracting all the data enclosed by "/p"
    pattern = r'/p(.*?)/p'
    results = re.findall(pattern, text)
    return results

##########################
### Conversational LLM ###
##########################

#OpenAI key in case of using a OpenAI model. Not used by default
openai.api_key = 'tu_clave_de_api_aquí'

#Function for generating the final answer for any GPT model of OpenAI
def generate_answer(memorable_data, conversation_history):

    #Adapt to the specific format of OpenAI models
    formatted_memories = ', '.join(memorable_data)
    messages = []
    memorable = """The next 5 pieces of data are relevant memorable data of the user 
    related to the actual context of the conversacion. Use it you consider necessary: 
    {}""".format(formatted_memories)
    messages.append({"role": "system", "content": memorable})

    for i, content in enumerate(conversation_history):
        role = "user" if i % 2 == 0 else "system"
        messages.append({"role": role, "content": content})
    
    #API request
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    #Extract the response
    model_response = response['choices'][0]['message']['content']
    if log == "full": print("Respuesta generada: ", model_response)
    
    return model_response

#Function in case of using a local model on Ollama. Used by default
def generate_answer2(memories, conversation_history):
    try:

        prompt = """Your are an empathic and affective conversational agent. Try to follow the flow of the user and show interest about what he talks about.
        The following is a set of memories related to the user, 
        as well as the history of the conversation with the user. 
        Generate the next turn of the conversation. Use the memories provided only if you find them relevant to the context.
        Do no base the conversation in these memories, just use it if you find them relevant to the current flow of the conversation.
        As this is an ongoing conversation, generate concise and pleasant answers to continue the conversation.
        The response you generate must be maximun 1 line.
        """

        #If database is empty and there are not memories
        if memories == []:
            formatted_memories = "There are not past memories currently, continue with the conversation"
        else:
            formatted_memories = ', '.join(memories)
        
        formatted_conversation = " ".join(conversation_history)
        full_prompt = f"Instruction: {prompt}, [start of the memories] {formatted_memories} [end of the memories] , [start of the conversation history] {formatted_conversation} [end of the conversation history] "
        
        #Used Llama3 model by dedault. Any conversational LLM on Ollama can be used
        response = ollama.generate(model="llama3:8b", prompt=full_prompt)
        return response["response"]
    except Exception as e:
        print(f"Error when generating the final answer with Ollama: {e}")

#################
### Atributes ###
#################

#Obtain the embedding of the memorable data
def get_embedding(text):

    #Tokenize the text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    #Averaging the outputs of the last layer to obtain a single embedding vector
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings.flatten().tolist()

#Obtain the oldest timestamp for normalization
def get_oldest_timestamp():
    oldest_data = memorable_data_collection.find_one(sort=[("timestamp", 1)])

    if oldest_data is not None:
        try:
            oldest_timestamp = datetime.strptime(oldest_data['timestamp'], "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        except ValueError:
            #Fallback if the timestamp does not include fractional seconds
            oldest_timestamp = datetime.strptime(oldest_data['timestamp'], "%Y-%m-%dT%H:%M:%S").timestamp()
    else:
        #Default to now if no data exists
        oldest_timestamp = datetime.now().timestamp()

    if log == "full": print("Oldest Timestamp: ", oldest_timestamp)

    return oldest_timestamp

#Obtain the max count for normalization
def get_max_count():

    #Get the entry with the highest count
    max_count_data = memorable_data_collection.find_one(sort=[("count", -1)])  

    #Get the max count but default to 1 if no data exists
    max_count = max_count_data['count'] if max_count_data is not None else 1

    if log == "full": print("Max Count: ", max_count)

    return max_count


##########################
### REMEMBER FUNCTIONS ###
##########################

#Function to compute the score for each memorable data
def calculate_score(data, embedding, current_embedding, timestamp, count, max_time_diff, max_count, current_time_seconds):

    #Current timestamp for normalization
    timestamp_seconds = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
    time_diff = (current_time_seconds - timestamp_seconds)

    #Decary rate based on the time difference with the oldest timestamp
    decay_rate = 1 / max_time_diff if max_time_diff > 0 else 1

    #Exponential decay normalization 
    time_score = np.exp((-decay_rate * time_diff) / 10)
    
    #Normalized count score
    count_score = count / max_count

    #Semantic similarity
    cosine_score = cosine_similarity([embedding], [current_embedding])[0][0]
    
    #Compute the weighted score, Weights can be adjusted
    weighted_score = 0.5 * cosine_score + 0.25 * time_score + 0.25 * count_score

    if log == "full": print(f"Data: {data}, Time Score: {time_score}, Count Score: {count_score}, Cosine Score: {cosine_score}, Weighted Score: {weighted_score}")
    
    return weighted_score

#Main function to retrieval the memorable data
def retrieve_memories(current_embedding, memory_size):

    #Obtain relevant atributes
    max_count = get_max_count()
    oldest_timestamp = get_oldest_timestamp()
    current_time_seconds = datetime.now().timestamp()

    if log == "full": print("Current timestampt: ", current_time_seconds)
    max_time_diff = current_time_seconds - oldest_timestamp
    if log == "full": print("Max time difference: ", max_time_diff)

    data = get_memorable_data()

    #If empty database, return an empy array to avoid problems
    if not data or data == []:
        return []
    
    #Compute all the scores
    if log == "full": print("SCORES FOR THE DATA START")
    scores = [(item, calculate_score(item["data"], np.array(item['embedding']), np.array(current_embedding), item['timestamp'], item['count'], max_time_diff, max_count, current_time_seconds)) for item in data]
    if log == "full": print("SCORES FOR THE DATA END")

    #Sort the data based on the scores and extract the amount of memory_size
    top_5 = sorted(scores, key=lambda x: x[1], reverse=True)[:memory_size]

    return top_5

##########################
### MEMORIZE FUNCTIONS ###
##########################

#Main function to memorize and store the memorable data of the conversation
def process_memory(user_turns, memorize_type):
    text = ". ".join(user_turns)

    #Called the API of the LLM in charge of extracting the memorable data
    memorable_items = memory_api(text, memorize_type)["response"]
    if log == "full": 
        print("Answer from the LLM in charge of extracting memories:")
        print(memorable_items)
        print("")

    #ReGeX function to extract and process the memorable data
    memorable_items = extract_data(memorable_items)

    if log == "full" or log == "basic": 
        if log == "basic":
            print("#####################")
            print("Extracted and memorized data:")
        else:
            print("Extracted Memories:")
        for item in memorable_items:
            print(item)
        if log == "basic":
            print("#####################")
    
    #Build data structure with the attributes related for storing the data
    memorable_data = [{'data': item, 'embedding': get_embedding(item.strip()), 'timestamp': datetime.now().isoformat(), 'count': 1} for item in memorable_items]
    
    if log == "full": print("START OF UPDATING/STORING MEMORIES")

    #Call the function for storing the data
    update_or_store_memorable_data(memorable_data)

    if log == "full": print("END OF UPDATING/STORING MEMORIES")

#Function for storing or updating the data
def update_or_store_memorable_data(memorable_data):

    #Obtain the memories from the database
    existing_memories = get_memorable_data()

    #For each of the extracted new memories, if it is the same memorie as one already in the database, update the atribute count.
    #If not, store the new memorie
    for new_data in memorable_data:
        try:
            new_embedding = np.array(new_data['embedding'])
            found = False

            #Compare with all the stored memories
            for existing in existing_memories:
                existing_embedding = np.array(existing['embedding'])

                #Check if the semantic similarity is enough to be the same memorie
                if cosine_similarity([new_embedding], [existing_embedding])[0][0] > 0.85:
                    memorable_data_collection.update_one({'_id': existing['_id']}, {'$inc': {'count': 1}, '$set': {'timestamp': datetime.now().isoformat()}})

                    if log == "full": print("Updated memorie data: ", existing["data"])
                    found = True
                    break
            #If not found enough similarity, store the new data
            if not found:
                memorable_data_collection.insert_one(new_data)

                if log == "full": print("Inserted memorable data: ",   new_data["data"])

        except Exception as e:
            print(f"Error when storing or updating new data: {e}")

#######################
### PARSE ARGUMENTS ###
#######################

def parse_arguments():
    global log
    parser = argparse.ArgumentParser(description="Your personal and afective Bot")
    parser.add_argument('--memory-interval', type=int, default=10,
                        help='Number of turns between memorization process')
    parser.add_argument('--log', type=str, default="no",
                        help='Specify the log functionality (no, basic, full)')
    parser.add_argument('--memory-size', type=int, default=5,
                        help='Size of the retrieval memories from long-term memory')
    parser.add_argument('--memorize-type', type=str, default="user",
                        help='Memorize based only on the turns of the user or based on the turns of the user and the bot')
    parser.add_argument('--context-size', type=int, default=5,
                        help='How many past turns of the conversation are used to compute the semantic similarity of the recall process')

    args = parser.parse_args()
    log = args.log
    return args


##########################

#Main method
def main():

    #Usefull atributes and parse arguments
    conversation_history = []
    turn_count = 0
    args = parse_arguments()
    memory_interval = args.memory_interval
    memory_size = args.memory_size
    memorize_type = args.memorize_type
    context_size = args.context_size

    print("Hi! I am your personal bot. What do you want to talk about today.")
    conversation_history.append("LLM: Hi! I am your personal bot.")

    #True loop till user exits
    while True:

        #Input of the user
        user_input = input("You: ")
        if user_input.lower() in ['salir', 'exit']:
            conversation_history.append("User: Bye!")
            print("Bye!")

            #Last memorization process
            left_turns = turn_count % memory_interval
            if log == "full": print("START LAST MEMORIZATION PROCESS")

            if memorize_type == "user":
                #Memorize only based on the turns of the user
                if log == "full": print("Memorizar los turnos:", conversation_history[-left_turns::2])
                process_memory(conversation_history[-left_turns::2], "user")
            else:
                #Memorize based on both the turns of the user and the turns of the bot
                if log == "full": print("Memorizar los turnos:", conversation_history[-left_turns:])
                process_memory(conversation_history[-left_turns:], "all")

            if log== "full": print("END LAST MEMORIZATION PROCESS")

            break
        
        #Append turn of the user
        conversation_history.append("User: "+user_input)
        memory_history = conversation_history

        #Use max 5 turns for having knowledge of the context of the conversation. Used in the recall process
        if len(conversation_history) > context_size:
            memory_history = conversation_history[-context_size:]

        ########################################################
        #Recall process management
        if log == "full": print("START RECALL PROCESS")

        current_embedding = get_embedding(" ".join(memory_history))
        relevant_memories = retrieve_memories(current_embedding, memory_size)

        #Extract textual data from the selected memories
        formatted_memories = [item[0]['data'] for item in relevant_memories]

        if log == "full" or log == "basic":
            print("#####################")
            print("Relevant memories:")
            if formatted_memories == []:
                print("Empty Memories Database")
            for memory, score in relevant_memories:
                print("Memory: {}, Score: {:.4f}".format(memory["data"], score))
            print("#####################")

        if log == "full": print("END RECALL PROCESS")
        ########################################################


        ########################################################
        #Memorization process management
        turn_count += 1

        #Memorize every memory interval
        if turn_count % memory_interval == 0:
            if log == "full": print("START MEMORIZATION PROCESS")

            if memorize_type == "user":
                #Memorize only based on the turns of the user
                if log == "full": print("Turns of the conversation to memorize:", conversation_history[-(2*memory_interval)::2])
                process_memory(conversation_history[-10::2], "user")
            else:
                #Memorize based on both the turns of the user and the turns of the bot
                if log == "full": print("Turns of the conversation to memorize:", conversation_history[-(memory_interval):])
                process_memory(conversation_history[-10:], "all")

            if log == "full": print("END MEMORIZATION PROCESS")
        ########################################################

        #Generate final answer with conversational LLM
        llm_response = generate_answer2(formatted_memories, conversation_history)
        print("Bot:", llm_response)

        #Append the answer of the LLM to the conversation history
        conversation_history.append("LLM: "+llm_response)

if __name__ == "__main__":
    main()
