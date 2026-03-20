from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import json


model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine(a,b):
    return np.dot(a,b)/(norm(a)*norm(b))

try:
    with open('embadded_scrappedData.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data["value"]:
            item['embedding'] = np.array(item['embedding'])
except FileNotFoundError:
    print("file not found")
except json.JSONDecodeError:
    print("json fucked up your file lil bro")

query_data = data['value']




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers = ["*"]
)

@app.get('/{query}')
async def resolve_query(query:str):
    final_matched_chunked = []
    original = query
    original_embadding = model.encode(original)


    if query_data:
        for a in query_data:
            a['similarity'] = cosine(original_embadding,(a['embedding']))
        query_data.sort(key=lambda x:x['similarity'], reverse = True)
        seen = set()
        unique_results = []
        for item in query_data:
            text_key = (str(item['text']), item['url'])
            if text_key not in seen:
                seen.add(text_key)
                unique_results.append({
                    'text': item['text'],
                    'url': item['url']
                })
        final_matched_chunked = [x for x in unique_results[:5]]

    else:
        final_matched_chunked = []
    


    return {
        'query': original,
        'results': final_matched_chunked,
        'count' : len(final_matched_chunked)
    }