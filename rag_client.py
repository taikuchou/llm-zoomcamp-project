import streamlit as st
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import ollama

index_name = "movie-questions"
model_name = "multi-qa-distilbert-cos-v1"
model = SentenceTransformer(model_name)
es_client = Elasticsearch('http://localhost:9200')


def hybrid_search_with_rrf(query, k=60):
    query_vector = model.encode(query)
    keyword_query = {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "summary", "genres"],
                    "type": "best_fields",
                    "boost": 0.5
                }
            },
        }
    }

    knn_query = {
        "field": "title_summary_genres_vector",
        "query_vector": query_vector,
        "k": 20,
        "num_candidates": 100,
        "boost": 0.5
    }
    # Hybrid search: combining both text and vector search (at least evaluating it)

    knn_results = es_client.search(
        index=index_name,
        body={
            "knn": knn_query,
            "size": 10
        }
    )['hits']['hits']

    keyword_results = es_client.search(
        index=index_name,
        body={
            "query": keyword_query,
            "size": 10
        }
    )['hits']['hits']

    rrf_scores = {}
    # Calculate RRF using vector search results
    for rank, hit in enumerate(knn_results):
        doc_id = hit['_id']
        rrf_scores[doc_id] = compute_rrf(rank + 1, k)

    # Adding keyword search result scores
    for rank, hit in enumerate(keyword_results):
        doc_id = hit['_id']
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += compute_rrf(rank + 1, k)
        else:
            rrf_scores[doc_id] = compute_rrf(rank + 1, k)

    # Sort RRF scores in descending order
    reranked_docs = sorted(
        rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Document re-ranking (1 point)
    # Get top-K documents by the score
    final_results = []
    for doc_id, score in reranked_docs[:5]:
        doc = es_client.get(index=index_name, id=doc_id)
        final_results.append(doc['_source'])

    return final_results


def compute_rrf(self, rank, k=60):
    """ Our own implementation of the relevance score """
    return 1 / (k + rank)


prompt_template = """
You're a movie consault. Answer the QUESTION based on the CONTEXT from our movies database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
title: {title}               
year: {year}     
summary: {summary}     
short_summary: {short_summary}     
genres: {genres}     
imdb_id: {imdb_id}     
runtime: {runtime}     
youtube_trailer: {youtube_trailer}     
rating: {rating}     
movie_poster: {movie_poster}     
director: {director}     
writers: {writers}     
cast: {cast}  
""".strip()


def build_prompt(query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


def rag(query, st):
    search_results = hybrid_search_with_rrf(query)
    prompt = build_prompt(query, search_results)
    stream = client.chat(
        model='gemma2:2b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )
    st.markdown("### Here you are:")
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
        yield chunk['message']['content']


ollama.pull('gemma2:2b')
client = ollama.Client(host='http://localhost:11434')
# Streamlit UI
st.title("Cinematic Advisory System")

# User input
query = st.text_input("Enter your question:")


# Button to trigger RAG process
if st.button("Go"):
    if query:
        # answer = rag(query, st)
        st.write_stream(rag(query, st))
    else:
        st.write("Kindly enter a valid question and try again.")
