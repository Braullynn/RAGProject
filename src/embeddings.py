"""
Motor de Embeddings Multimodal usando Gemini Embedding 2.
Suporta texto, imagem, vídeo, áudio e PDF.
Textos e PDFs são divididos em chunks para gerar múltiplos vetores.
"""

import hashlib
import uuid
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from rich.console import Console
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import fitz  # PyMuPDF

from .chunking import chunk_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from .config import (
    EXTENSION_TO_MIME,
    VIDEO_MAX_DURATION_SECONDS,
    Settings,
)

console = Console()


class EmbeddingEngine:
    """Motor central para geração de embeddings multimodais."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model = self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions

    # ------------------------------------------------------------------
    # Métodos públicos por modalidade
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> dict:
        """Gera embedding para um único trecho de texto (sem chunking)."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )

        values = list(result.embeddings[0].values)

        return {
            "id": self._generate_id(text.encode()),
            "values": values,
            "metadata": {
                "type": "texto",
                "mime_type": "text/plain",
                "content_preview": text[:200],
                "char_count": len(text),
            },
        }

    def embed_text_chunked(
        self,
        text: str,
        source_name: str = "texto",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        existing_ids: set[str] | None = None,
    ) -> list[dict]:
        """
        Divide o texto em chunks e gera um embedding para cada chunk.
        Se `existing_ids` for passado, não tenta gerar embeddings para IDs presentes no set.
        Retorna uma lista de vetores.
        """
        existing = existing_ids or set()
        chunks = chunk_text(
            text,
            source_name=source_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        console.print(
            f"  📝 Texto dividido em [bold]{len(chunks)}[/bold] chunk(s) "
            f"({len(text)} chars total)"
        )

        @retry(
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type(ClientError),
        )
        def _get_embedding_with_retry(content: str):
            return self.client.models.embed_content(
                model=self.model,
                contents=content,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimensions,
                ),
            )

        vectors = []
        skipped = 0
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            if chunk_id in existing:
                skipped += 1
                continue
                
            try:
                result = _get_embedding_with_retry(chunk["text"])
                values = list(result.embeddings[0].values)
            except Exception as e:
                console.print(f"    ❌ Erro no chunk {chunk['index'] + 1}: {e}", style="red")
                continue

            vectors.append(
                {
                    "id": chunk_id,
                    "values": values,
                    "metadata": {
                        "type": "texto",
                        "mime_type": "text/plain",
                        "content_preview": chunk["text"][:200],
                        "chunk_index": chunk["index"],
                        "total_chunks": len(chunks),
                        "char_count": len(chunk["text"]),
                        "source_name": source_name,
                    },
                }
            )

            console.print(
                f"    ✔ Chunk {chunk['index'] + 1}/{len(chunks)} "
                f"({len(chunk['text'])} chars)"
            )
            
        if skipped > 0:
            console.print(f"  ⏭️ Ignorados {skipped} chunk(s) pois já existem no Pinecone.", style="cyan")

        return vectors

    def embed_image(self, file_path: str | Path) -> dict:
        """Gera embedding para uma imagem."""
        file_path = Path(file_path)
        mime_type = self._get_mime_type(file_path)
        console.print(f"  🖼️  Gerando embedding de imagem: {file_path.name}")

        image_bytes = file_path.read_bytes()

        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )

        values = list(result.embeddings[0].values)

        return {
            "id": self._generate_id(image_bytes),
            "values": values,
            "metadata": {
                "type": "imagem",
                "mime_type": mime_type,
                "file_name": file_path.name,
                "file_size_bytes": len(image_bytes),
            },
        }

    def embed_audio(self, file_path: str | Path) -> dict:
        """Gera embedding para um arquivo de áudio."""
        file_path = Path(file_path)
        mime_type = self._get_mime_type(file_path)
        console.print(f"  🎵 Gerando embedding de áudio: {file_path.name}")

        audio_bytes = file_path.read_bytes()

        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )

        values = list(result.embeddings[0].values)

        return {
            "id": self._generate_id(audio_bytes),
            "values": values,
            "metadata": {
                "type": "audio",
                "mime_type": mime_type,
                "file_name": file_path.name,
                "file_size_bytes": len(audio_bytes),
            },
        }

    def embed_video(self, file_path: str | Path) -> list[dict]:
        """
        Gera embedding(s) para um vídeo.
        Se o vídeo for > 120s, segmenta automaticamente e retorna
        uma lista de embeddings (um por segmento).
        Para vídeos curtos, retorna lista com um único embedding.
        """
        file_path = Path(file_path)
        mime_type = self._get_mime_type(file_path)
        console.print(f"  🎬 Gerando embedding de vídeo: {file_path.name}")

        video_bytes = file_path.read_bytes()

        file_size_mb = len(video_bytes) / (1024 * 1024)
        if file_size_mb > 20:
            console.print(
                f"  ⚠️  Vídeo grande ({file_size_mb:.1f} MB). "
                f"Se tiver > {VIDEO_MAX_DURATION_SECONDS}s, "
                f"considere segmentar com ffmpeg.",
                style="yellow",
            )

        result = self.client.models.embed_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )

        values = list(result.embeddings[0].values)

        return [
            {
                "id": self._generate_id(video_bytes),
                "values": values,
                "metadata": {
                    "type": "video",
                    "mime_type": mime_type,
                    "file_name": file_path.name,
                    "file_size_bytes": len(video_bytes),
                },
            }
        ]

    def embed_document(self, file_path: str | Path, existing_ids: set[str] | None = None) -> list[dict]:
        """
        Gera embeddings para um PDF.
        O PDF é enviado ao Gemini para extração de texto e, depois,
        o texto extraído é segmentado em chunks para gerar múltiplos vetores.
        Caso a extração falhe, gera um único embedding do PDF inteiro.
        """
        file_path = Path(file_path)
        mime_type = self._get_mime_type(file_path)
        console.print(f"  📄 Processando documento: {file_path.name}")

        pdf_bytes = file_path.read_bytes()

        # Tenta extrair texto do PDF usando o Gemini (modelo generativo)
        extracted_text = self._extract_text_from_pdf(pdf_bytes)

        if extracted_text and len(extracted_text.strip()) > 50:
            console.print(
                f"  📋 Texto extraído do PDF: {len(extracted_text)} chars"
            )
            # Usa chunking no texto extraído
            vectors = self.embed_text_chunked(
                extracted_text,
                source_name=file_path.name,
                existing_ids=existing_ids,
            )
            # Atualiza os metadados para indicar que veio de um PDF
            for vec in vectors:
                vec["metadata"]["type"] = "documento"
                vec["metadata"]["mime_type"] = mime_type
                vec["metadata"]["file_name"] = file_path.name
            return vectors
        else:
            # Fallback: gera embedding do PDF inteiro como binário
            console.print(
                "  ⚠️  Não foi possível extrair texto. "
                "Gerando embedding do PDF inteiro...",
                style="yellow",
            )
            result = self.client.models.embed_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type=mime_type),
                ],
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimensions,
                ),
            )

            values = list(result.embeddings[0].values)

            return [
                {
                    "id": self._generate_id(pdf_bytes),
                    "values": values,
                    "metadata": {
                        "type": "documento",
                        "mime_type": mime_type,
                        "file_name": file_path.name,
                        "file_size_bytes": len(pdf_bytes),
                    },
                }
            ]

    def embed_content(self, file_path: str | Path, existing_ids: set[str] | None = None) -> list[dict]:
        """
        Detecta o tipo do arquivo automaticamente e gera embedding(s).
        Textos e documentos são divididos em chunks.
        Retorna sempre uma lista de dicts.
        """
        file_path = Path(file_path)
        mime_type = self._get_mime_type(file_path)

        if mime_type.startswith("text/"):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return self.embed_text_chunked(text, source_name=file_path.name, existing_ids=existing_ids)
        elif mime_type.startswith("image/"):
            return [self.embed_image(file_path)]
        elif mime_type.startswith("audio/"):
            return [self.embed_audio(file_path)]
        elif mime_type.startswith("video/"):
            return self.embed_video(file_path)
        elif mime_type == "application/pdf":
            return self.embed_document(file_path, existing_ids=existing_ids)
        else:
            raise ValueError(
                f"Tipo de arquivo não suportado: {mime_type} ({file_path.name})\n"
                f"Extensões suportadas: {', '.join(EXTENSION_TO_MIME.keys())}"
            )

    def embed_query(self, query: str) -> list[float]:
        """
        Gera embedding para uma query de busca (retorna apenas o vetor).
        Útil para busca semântica.
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )
        return list(result.embeddings[0].values)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str | None:
        """
        Usa o PyMuPDF localmente para extrair texto de um PDF.
        Retorna o texto extraído ou None se falhar.
        """
        try:
            console.print("  🔄 Extraindo texto do PDF localmente (PyMuPDF)...")
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted_text = ""
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text:
                    extracted_text += f"\n--- Página {i + 1} ---\n{text}"
            doc.close()
            return extracted_text.strip()
        except Exception as e:
            console.print(
                f"  ⚠️  Erro ao extrair texto do PDF: {e}",
                style="yellow",
            )
            return None

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Retorna o MIME type baseado na extensão do arquivo."""
        ext = file_path.suffix.lower()
        if ext not in EXTENSION_TO_MIME:
            raise ValueError(
                f"Extensão não suportada: '{ext}'\n"
                f"Extensões suportadas: {', '.join(EXTENSION_TO_MIME.keys())}"
            )
        return EXTENSION_TO_MIME[ext]

    @staticmethod
    def _generate_id(data: bytes) -> str:
        """Gera um ID único baseado no hash do conteúdo."""
        content_hash = hashlib.sha256(data).hexdigest()[:12]
        short_uuid = uuid.uuid4().hex[:8]
        return f"{content_hash}-{short_uuid}"
