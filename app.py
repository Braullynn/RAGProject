import streamlit as st
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

# Accept user input
if prompt := st.chat_input("Qual a sua dúvida?"):
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
