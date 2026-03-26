import streamlit as st
import time
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from src.config import Settings
from src.search import SearchEngine

st.set_page_config(
    page_title="RAG Multimodal Chat",
    page_icon="🧠",
    layout="wide",
)

@st.cache_resource
def get_search_engine():
    return SearchEngine(Settings())

@st.cache_resource
def get_gemini_client():
    settings = Settings()
    return genai.Client(api_key=settings.gemini_api_key)

st.title("🧠 Naruto RPG")
st.markdown("Consulte as regras do RPG de Naruto.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Fontes Utilizadas"):
                for source in message["sources"]:
                    st.caption(f"📄 {source['file']} (Score: {source['score']:.2f})")
                    st.markdown(f"> {source['text']}")

import time

# Rate limiting inicialization
if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []
if "block_until" not in st.session_state:
    st.session_state.block_until = 0

current_time = time.time()

# Limpar requests antigos (> 60s)
st.session_state.request_timestamps = [t for t in st.session_state.request_timestamps if current_time - t < 60]

# Liberação natural do bloqueio após o tempo acabar
if st.session_state.block_until > 0 and current_time >= st.session_state.block_until:
    st.session_state.block_until = 0
    st.session_state.request_timestamps = []

# Se estiver bloqueado e ocorrer um refresh no navegador no meio da contagem
if st.session_state.block_until > current_time:
    countdown_placeholder = st.empty()
    remaining = int(st.session_state.block_until - current_time)
    for i in range(remaining, 0, -1):
        countdown_placeholder.warning(f"⏳ **Limite de Proteção:** Você enviou requisições demais (10/minuto). Aguarde **{i} segundos** para liberação do sistema.")
        time.sleep(1)
    countdown_placeholder.empty()
    st.session_state.block_until = 0
    st.session_state.request_timestamps = []
    st.rerun()

# Accept user input
if prompt := st.chat_input("Qual a sua dúvida?"):
    # Add new timestamp
    st.session_state.request_timestamps.append(time.time())
    
    # Check rate limit
    if len(st.session_state.request_timestamps) >= 10:
        st.session_state.block_until = time.time() + 60
        countdown_placeholder = st.empty()
        for i in range(60, 0, -1):
            countdown_placeholder.warning(f"⏳ **Limite de Proteção:** Você enviou 10 requisições em menos de 1 minuto. Aguarde **{i} segundos** para liberação do sistema.")
            time.sleep(1)
        countdown_placeholder.empty()
        st.session_state.block_until = 0
        st.session_state.request_timestamps = []
        st.rerun()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("🔍 Buscando contexto no Pinecone..."):
                engine = get_search_engine()
                # Buscar contexto baseado na pergunta
                results = engine.search_by_text(prompt, top_k=6)
            
            # Preparar contexto para o Gemini
            context_parts = []
            sources = []
            for res in results:
                meta = res.get("metadata", {})
                score = res.get("score", 0.0)
                text = meta.get("content_preview", "")
                file_name = meta.get("file_name", meta.get("source_name", "Desconhecido"))
                
                if text:
                    context_parts.append(f"[Arquivo: {file_name}]\nTrecho: {text}")
                    sources.append({
                        "file": file_name,
                        "score": score,
                        "text": text
                    })
            
            context_str = "\n\n".join(context_parts)
            
            # System prompt instructions
            sys_instruction = (
                "Você é uma assistente virtual chamada Lara, respondendo perguntas com base "
                "exclusivamente nos trechos de documentos (contexto) fornecidos abaixo. "
                "Seja claro, respeitosa e educada. Fale sempre em Português do Brasil.\n"
                "Se a resposta não estiver no contexto, diga que não encontrou a informação.\n\n"
                "CONTEXTO EXTRAÍDO DO PINECONE:\n"
                f"{context_str}"
            )

            with st.spinner("💭 Gerando resposta (Gemini 2.5 Flash)..."):
                client = get_gemini_client()
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        temperature=0.3, # Baixa temperatura para RAG (respostas factuais)
                    )
                )
                
                full_response = response.text
                message_placeholder.markdown(full_response)
                
                if sources:
                    with st.expander("Fontes Utilizadas"):
                        for source in sources:
                            st.caption(f"📄 {source['file']} (Score: {source['score']:.2f})")
                            st.markdown(f"> {source['text']}")
                            
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
        except ClientError as e:
            if "429" in str(e):
                st.warning("⏳ **Cota atingida!** O nosso limite de uso diário da API gratuita do Google esgotou (Erro 429). Por favor, tente perguntar novamente amanhã quando a cota resetar.")
            else:
                st.error("Ocorreu um erro de comunicação com a inteligência artificial. Tente novamente mais tarde.")
        except Exception as e:
            st.error("Ocorreu um erro interno da aplicação.")
