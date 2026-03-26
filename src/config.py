"""
Módulo de configuração do projeto RAG Multimodal.
Carrega variáveis de ambiente e valida chaves obrigatórias.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Carrega o .env do diretório raiz do projeto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# Tipos MIME suportados por modalidade
SUPPORTED_MIME_TYPES: dict[str, list[str]] = {
    "texto": [
        "text/plain",
    ],
    "imagem": [
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
        "image/bmp",
    ],
    "video": [
        "video/mp4",
        "video/mpeg",
        "video/webm",
        "video/avi",
        "video/quicktime",
    ],
    "audio": [
        "audio/mpeg",
        "audio/wav",
        "audio/ogg",
        "audio/flac",
        "audio/aac",
    ],
    "documento": [
        "application/pdf",
    ],
}

# Mapeamento extensão → MIME type
EXTENSION_TO_MIME: dict[str, str] = {
    # Texto
    ".txt": "text/plain",
    ".md": "text/plain",
    ".csv": "text/plain",
    ".json": "text/plain",
    # Imagem
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    # Vídeo
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".webm": "video/webm",
    ".avi": "video/avi",
    ".mov": "video/quicktime",
    # Áudio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    # Documento
    ".pdf": "application/pdf",
}

# Limite de duração de vídeo em segundos (Gemini Embedding 2)
VIDEO_MAX_DURATION_SECONDS = 120

# Limite máximo de tokens de entrada
MAX_INPUT_TOKENS = 8192


@dataclass
class Settings:
    """Configurações centrais do projeto."""

    # Google Gemini
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Pinecone
    pinecone_api_key: str = field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", "")
    )
    pinecone_index_name: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "rag-multimodal")
    )
    pinecone_cloud: str = field(
        default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws")
    )
    pinecone_region: str = field(
        default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1")
    )

    # Modelo de Embedding
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "gemini-embedding-2-preview"
        )
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    )

    def validate(self) -> bool:
        """Valida que todas as chaves obrigatórias estão preenchidas."""
        errors: list[str] = []

        if not self.gemini_api_key or self.gemini_api_key == "SUA_CHAVE_GEMINI_AQUI":
            errors.append(
                "❌ GEMINI_API_KEY não configurada. Edite o arquivo .env"
            )

        if (
            not self.pinecone_api_key
            or self.pinecone_api_key == "SUA_CHAVE_PINECONE_AQUI"
        ):
            errors.append(
                "❌ PINECONE_API_KEY não configurada. Edite o arquivo .env"
            )

        if errors:
            for err in errors:
                print(err, file=sys.stderr)
            return False

        return True

    def __str__(self) -> str:
        return (
            f"⚙️  Configurações RAG Multimodal\n"
            f"   Modelo:     {self.embedding_model}\n"
            f"   Dimensões:  {self.embedding_dimensions}\n"
            f"   Pinecone:   {self.pinecone_index_name} "
            f"({self.pinecone_cloud}/{self.pinecone_region})\n"
            f"   Gemini Key: {'✅ Configurada' if self.gemini_api_key and self.gemini_api_key != 'SUA_CHAVE_GEMINI_AQUI' else '❌ Pendente'}\n"
            f"   Pinecone Key: {'✅ Configurada' if self.pinecone_api_key and self.pinecone_api_key != 'SUA_CHAVE_PINECONE_AQUI' else '❌ Pendente'}"
        )
