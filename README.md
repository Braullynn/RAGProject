# RAG Multimodal Project (Gemini 2.5 + Pinecone)

Este repositório implementa um poderoso sistema de Retrieval-Augmented Generation (RAG) Multimodal. Ele processa documentos diversos (textos e PDFs) utilizando o modelo **Gemini Embedding 2** para buscar os dados semânticos, e os armazena no banco vetorial **Pinecone**.

O sistema conta com um recurso de **Retomada Automática Inteligente (Resumable Ingest)**. Ele evita processar arquivos já salvos lendo `hashes` determinísticos dos arquivos locais contra o banco de vetores antes da submissão na API. Uma interface em **Streamlit** e **Gemini Flash** permite que você interaja e converse visualmente com seus dados.

## ⚙️ Pré-requisitos
- Python 3.10+
- [Chave de API do Google AI Studio (Gemini)](https://aistudio.google.com/)
- [Chave de API do Pinecone](https://app.pinecone.io/) (Índice Serverless com Dimensão 1536)

## 🚀 Instalação e Configuração
1. **Instale as dependências**
   Abra o seu terminal na raiz do projeto e instale todos os módulos:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure suas Credenciais**
   Renomeie ou crie o arquivo `.env` a partir do `env.example` protegendo suas chaves. Nunca comite seu `.env`!
   ```env
   GEMINI_API_KEY=sua-chave-aqui
   PINECONE_API_KEY=sua-chave-aqui
   PINECONE_INDEX_NAME=rag-multimodal
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   EMBEDDING_MODEL=gemini-embedding-2-preview
   EMBEDDING_DIMENSIONS=1536
   ```

## 🧠 Como usar a Interface Web (Chatbot)
Toda a interação de perguntas e busca de recortes nos seus arquivos PDF/TXT roda pelo app principal! Basta digitar o comando abaixo:
```bash
streamlit run app.py
```
Seu navegador abrirá em `localhost:8501`.

### Extra: Ingestão de Documentos em Batch via Terminal
Para processar e inserir (submeter) uma pasta inteira no Pinecone, coloque seus PDFs/TXTs na subpasta e rode:
```bash
python main.py ingest "data/documentos"
```
