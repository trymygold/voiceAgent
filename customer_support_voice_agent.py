from typing import List, Dict, Optional
from pathlib import Path
import os
from firecrawl import FirecrawlApp
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding
from agents import Agent, Runner
from openai import AsyncOpenAI
import tempfile
import uuid
from datetime import datetime
import time
import streamlit as st
from dotenv import load_dotenv
import asyncio

load_dotenv()

def init_session_state():
    # Nishanth: Update these default values with your NEW keys
    defaults = {
        "initialized": False,
        "qdrant_url": "YOUR_QDRANT_CLUSTER_URL", 
        "qdrant_api_key": "YOUR_QDRANT_API_KEY",
        "firecrawl_api_key": "YOUR_FIRECRAWL_KEY",
        "openai_api_key": "YOUR_NEW_OPENAI_KEY",
        "doc_url": "https://jewels-ai.online", # Your startup domain
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "tts_agent": None,
        "selected_voice": "coral"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def sidebar_config():
    with st.sidebar:
        st.title("💎 Jewels-AI Admin")
        st.markdown("---")
        
        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
            type="password"
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password"
        )
        st.session_state.firecrawl_api_key = st.text_input(
            "Firecrawl API Key",
            value=st.session_state.firecrawl_api_key,
            type="password"
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password"
        )
        
        st.markdown("---")
        st.session_state.doc_url = st.text_input(
            "Jewellery Knowledge Base (URL)",
            value=st.session_state.doc_url,
            placeholder="https://docs.example.com"
        )
        
        st.markdown("---")
        st.markdown("### 🎤 Voice Settings")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox(
            "Agent Voice",
            options=voices,
            index=voices.index(st.session_state.selected_voice)
        )
        
        if st.button("Sync Knowledge Base", type="primary"):
            if all([st.session_state.qdrant_url, st.session_state.qdrant_api_key, 
                    st.session_state.firecrawl_api_key, st.session_state.openai_api_key, 
                    st.session_state.doc_url]):
                
                with st.spinner("Initializing AI Agents..."):
                    try:
                        client, embedding_model = setup_qdrant_collection(
                            st.session_state.qdrant_url,
                            st.session_state.qdrant_api_key
                        )
                        st.session_state.client = client
                        st.session_state.embedding_model = embedding_model
                        
                        pages = crawl_documentation(
                            st.session_state.firecrawl_api_key,
                            st.session_state.doc_url
                        )
                        
                        store_embeddings(client, embedding_model, pages, "jewels_db")
                        
                        p_agent, t_agent = setup_agents(st.session_state.openai_api_key)
                        st.session_state.processor_agent = p_agent
                        st.session_state.tts_agent = t_agent
                        
                        st.session_state.setup_complete = True
                        st.success("✅ Jewels-AI is now Live!")
                    except Exception as e:
                        st.error(f"Setup Error: {str(e)}")
            else:
                st.error("Fill all API fields first!")

def setup_qdrant_collection(url, key, collection_name="jewels_db"):
    client = QdrantClient(url=url, api_key=key)
    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(test_embedding), distance=Distance.COSINE)
        )
    except Exception as e:
        if "already exists" not in str(e): raise e
    return client, embedding_model

def crawl_documentation(api_key, url):
    firecrawl = FirecrawlApp(api_key=api_key)
    response = firecrawl.crawl_url(url, params={'limit': 5, 'scrapeOptions': {'formats': ['markdown']}})
    pages = []
    
    for page in response.get('data', []):
        pages.append({
            "content": page.get('markdown', ''),
            "url": page.get('metadata', {}).get('sourceURL', ''),
            "metadata": {
                "title": page.get('metadata', {}).get('title', 'Jewellery Info'),
                "crawl_date": datetime.now().isoformat()
            }
        })
    return pages

def store_embeddings(client, model, pages, collection_name):
    for page in pages:
        embedding = list(model.embed([page["content"]]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"content": page["content"], "url": page["url"], **page["metadata"]}
            )]
        )

def setup_agents(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Customized Persona for Nishanth's startup
    processor_agent = Agent(
        name="Jewels-AI Expert",
        instructions="""You are a luxury jewellery consultant for Jewels-AI. 
        1. Use the provided jewellery documentation to answer queries.
        2. Speak elegantly about diamonds, gold purity, and AR try-on features.
        3. Be concise and conversational, as your output will be spoken.
        4. Mention the specific collection or URL if referencing a product.""",
        model="gpt-4o"
    )

    tts_agent = Agent(
        name="Luxury Voice Agent",
        instructions="""Refine the text for a smooth, high-end audio delivery. 
        Add natural pauses and ensure technical jewellery terms are pronounced clearly.""",
        model="gpt-4o-mini-tts"
    )
    return processor_agent, tts_agent

async def process_query(query, client, model, p_agent, t_agent, openai_key):
    query_embedding = list(model.embed([query]))[0]
    search_response = client.query_points(
        collection_name="jewels_db",
        query=query_embedding.tolist(),
        limit=3
    )
    
    context = "Customer Question: " + query + "\n\nDocumentation Context:\n"
    sources = []
    for r in search_response.points:
        context += f"Source ({r.payload['url']}): {r.payload['content']}\n"
        sources.append(r.payload['url'])
    
    p_result = await Runner.run(p_agent, context)
    text_out = p_result.final_output
    
    t_result = await Runner.run(t_agent, text_out)
    
    async_client = AsyncOpenAI(api_key=openai_key)
    audio = await async_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=st.session_state.selected_voice,
        input=text_out,
        response_format="mp3"
    )
    
    audio_path = os.path.join(tempfile.gettempdir(), f"jewel_{uuid.uuid4()}.mp3")
    with open(audio_path, "wb") as f: f.write(audio.content)
    
    return {"text": text_out, "audio": audio_path, "sources": list(set(sources))}

def main():
    st.set_page_config(page_title="Jewels-AI Voice Concierge", page_icon="💎")
    init_session_state()
    sidebar_config()
    
    st.title("🎙️ Jewels-AI Voice Concierge")
    st.info("Ask about your jewellery collections or the AR virtual try-on technology.")

    query = st.text_input("How can I assist you today?", placeholder="e.g. Tell me about the diamond earrings collection", disabled=not st.session_state.setup_complete)

    if query and st.session_state.setup_complete:
        with st.spinner("Consulting our experts..."):
            res = asyncio.run(process_query(
                query, st.session_state.client, st.session_state.embedding_model,
                st.session_state.processor_agent, st.session_state.tts_agent,
                st.session_state.openai_api_key
            ))
            
            st.markdown("### 💍 Expert Advice")
            st.write(res["text"])
            st.audio(res["audio"], format="audio/mp3")
            
            with st.expander("View Sources"):
                for s in res["sources"]: st.write(f"- {s}")

if __name__ == "__main__":
    main()