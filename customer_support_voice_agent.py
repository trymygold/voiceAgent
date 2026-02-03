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

# Load variables from a .env file if it exists
load_dotenv()

def init_session_state():
    """Initializes the session state with Jewels-AI specific defaults."""
    defaults = {
        "initialized": False,
        "qdrant_url": "YOUR_QDRANT_CLUSTER_URL", 
        "qdrant_api_key": "YOUR_NEW_QDRANT_KEY",
        "firecrawl_api_key": "YOUR_NEW_FIRECRAWL_KEY",
        "openai_api_key": "YOUR_NEW_OPENAI_KEY",
        "doc_url": "https://jewels-ai.online", 
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
    """Handles the sidebar configuration and system initialization."""
    with st.sidebar:
        st.title("💎 Jewels-AI Admin")
        st.markdown("---")
        
        # Configuration Inputs
        st.session_state.qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url)
        st.session_state.qdrant_api_key = st.text_input("Qdrant API Key", value=st.session_state.qdrant_api_key, type="password")
        st.session_state.firecrawl_api_key = st.text_input("Firecrawl API Key", value=st.session_state.firecrawl_api_key, type="password")
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
        
        st.markdown("---")
        st.session_state.doc_url = st.text_input(
            "Jewellery Knowledge Base (URL)",
            value=st.session_state.doc_url,
            placeholder="https://jewels-ai.online"
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
                
                with st.spinner("Syncing data to Jewels-AI DB..."):
                    try:
                        # 1. Setup Vector DB
                        client, embedding_model = setup_qdrant_collection(
                            st.session_state.qdrant_url,
                            st.session_state.qdrant_api_key
                        )
                        st.session_state.client = client
                        st.session_state.embedding_model = embedding_model
                        
                        # 2. Crawl Website
                        pages = crawl_documentation(
                            st.session_state.firecrawl_api_key,
                            st.session_state.doc_url
                        )
                        
                        # 3. Store in Qdrant
                        store_embeddings(client, embedding_model, pages, "jewels_db")
                        
                        # 4. Initialize Agents
                        p_agent, t_agent = setup_agents(st.session_state.openai_api_key)
                        st.session_state.processor_agent = p_agent
                        st.session_state.tts_agent = t_agent
                        
                        st.session_state.setup_complete = True
                        st.success("✅ Jewels-AI is now Live!")
                    except Exception as e:
                        st.error(f"Setup Error: {str(e)}")
            else:
                st.error("Please provide all API keys to initialize.")

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
    # Limited to 5 pages for efficiency
    response = firecrawl.crawl_url(url, params={'limit': 5, 'scrapeOptions': {'formats': ['markdown']}})
    pages = []
    
    for page in response.get('data', []):
        pages.append({
            "content": page.get('markdown', ''),
            "url": page.get('metadata', {}).get('sourceURL', ''),
            "metadata": {
                "title": page.get('metadata', {}).get('title', 'Product Info'),
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
    
    # Custom Persona for Nishanth's startup
    processor_agent = Agent(
        name="Jewels-AI Consultant",
        instructions="""You are a high-end luxury jewellery consultant for Jewels-AI. 
        1. Use the provided context to answer questions about products, diamonds, or gold.
        2. Mention our Virtual AR Try-on feature whenever relevant.
        3. Keep the tone sophisticated, helpful, and concise.
        4. Since this will be spoken, avoid long lists or complex markdown.""",
        model="gpt-4o"
    )

    tts_agent = Agent(
        name="Voice Stylist",
        instructions="""Review the text and optimize it for audio. 
        Ensure technical terms are clear and the pacing feels like a natural human conversation.""",
        model="gpt-4o-mini-tts"
    )
    return processor_agent, tts_agent

async def process_query(query, client, model, p_agent, t_agent, openai_key):
    # Retrieve relevant data from Vector DB
    query_embedding = list(model.embed([query]))[0]
    search_response = client.query_points(
        collection_name="jewels_db",
        query=query_embedding.tolist(),
        limit=3
    )
    
    context = f"User Query: {query}\n\nDocumentation Context:\n"
    sources = []
    for r in search_response.points:
        context += f"Source ({r.payload['url']}): {r.payload['content']}\n"
        sources.append(r.payload['url'])
    
    # Generate text response
    p_result = await Runner.run(p_agent, context)
    text_out = p_result.final_output
    
    # Refine for TTS
    t_result = await Runner.run(t_agent, text_out)
    
    # Generate Audio
    async_client = AsyncOpenAI(api_key=openai_key)
    audio = await async_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=st.session_state.selected_voice,
        input=text_out,
        response_format="mp3"
    )
    
    audio_path = os.path.join(tempfile.gettempdir(), f"jewel_response_{uuid.uuid4()}.mp3")
    with open(audio_path, "wb") as f: f.write(audio.content)
    
    return {"text": text_out, "audio": audio_path, "sources": list(set(sources))}

def main():
    st.set_page_config(page_title="Jewels-AI Voice Concierge", page_icon="💎")
    init_session_state()
    sidebar_config()
    
    st.title("🎙️ Jewels-AI Voice Concierge")
    st.info("Ask about our collections, gold rates, or how to use the Virtual Try-on.")

    query = st.text_input(
        "How can I help you today?", 
        placeholder="e.g., Show me your diamond earring collections.", 
        disabled=not st.session_state.setup_complete
    )

    if query and st.session_state.setup_complete:
        with st.spinner("Connecting to a consultant..."):
            try:
                res = asyncio.run(process_query(
                    query, st.session_state.client, st.session_state.embedding_model,
                    st.session_state.processor_agent, st.session_state.tts_agent,
                    st.session_state.openai_api_key
                ))
                
                st.markdown("### 💍 Consultant Response")
                st.write(res["text"])
                st.audio(res["audio"], format="audio/mp3")
                
                with st.expander("Reference Sources"):
                    for s in res["sources"]: st.write(f"- {s}")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()