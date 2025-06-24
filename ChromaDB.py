import chromadb
from chromadb.utils import embedding_functions
import wikipediaapi
from sentence_transformers import SentenceTransformer

chromaClient=chromadb.PersistentClient(path="./chromaDB")
try:
    model=SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded")
except Exception as e:
    print(f"Error {e}") 

sentence_transformer_ef=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
coll_name="WikipediaArticles"
try:
    collection=chromaClient.get_or_create_collection(name=coll_name,embedding_function=sentence_transformer_ef)
    print("Collection created")
except Exception as e:
    print(f"Error {e}")    

wiki_wiki=wikipediaapi.Wikipedia(user_agent='VECTORDB (330amina#gmail.com)',language='en')

article_titles=["Artificial Intelligence","Machine Learning","Deep Learning","Natural Language Processing","Neural Network",
                "Reinforcement Learning","Data Science","Computer Vision","Data Mining","Computer Science"]

print("Fetching articles")
docs=[]
metadata=[]
idstoadd=[]

for i,t in enumerate(article_titles):
    try:
        page=wiki_wiki.page(t)
        if page.exists():
            docs.append(page.summary)
            metadata.append({"title":t})
            idstoadd.append(str(i))
            print(f"Article found")
        else:
            print(f"Article not found")    
    except Exception as e:
        print(f"Error {e}")        

try:
     existing=collection.get(ids=idstoadd)["ids"]
     if existing:
         collection.delete(ids=existing)
         print(f"Deleted existing ids")
     collection.add(documents=docs,metadatas=metadata,ids=idstoadd)    
     print(f"Data added to collection")
except Exception as e:
    print(f"Error {e}")

while True:
   text=input("Enter your query or exit to stop:")
   if text.lower()=='exit':
       break
   print(f"Top match based on cosine similarity")
   try:
        results=collection.query(
        query_texts=[text],
        n_results=1,
        include=['documents','metadatas','distances'])
        if results and results['documents'] and results['documents'][0]:
         print(f"Title: {results['metadatas'][0][0]['title']}")
         print(f"Summary: {results['documents'][0][0]}")
         print(f"Cosine distance: {results['distances'][0][0]}")
        else:
         print(f"No matching data found")
   except Exception as e:
    print(f"Error {e}")