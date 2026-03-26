"""
Módulo de Chunking (segmentação) de texto.
Divide textos grandes em pedaços menores para gerar
embeddings mais granulares e melhorar a qualidade do RAG.
"""

from __future__ import annotations

import hashlib
import re
import uuid


# Tamanho padrão de cada chunk em caracteres
DEFAULT_CHUNK_SIZE = 1000

# Sobreposição entre chunks para manter contexto nas bordas
DEFAULT_CHUNK_OVERLAP = 200


def chunk_text(
    text: str,
    source_name: str = "texto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Divide um texto em pedaços menores com sobreposição.

    Estratégia:
    1. Primeiro tenta dividir por parágrafos/seções
    2. Se um parágrafo for maior que chunk_size, divide por sentenças
    3. Mantém sobreposição entre chunks para preservar contexto

    Args:
        text: Texto completo a ser dividido.
        chunk_size: Tamanho máximo de cada chunk em caracteres.
        chunk_overlap: Sobreposição entre chunks consecutivos.

    Returns:
        Lista de dicts com 'text', 'index' e 'chunk_id'.
    """
    if not text or not text.strip():
        return []

    # Limpa espaços excessivos
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Se o texto inteiro cabe em um chunk, retorna como está
    if len(text) <= chunk_size:
        return [
            {
                "text": text,
                "index": 0,
                "chunk_id": _generate_chunk_id(source_name, 0, text),
            }
        ]

    # Divide por parágrafos primeiro (dupla quebra de linha)
    paragraphs = re.split(r"\n\n+", text)

    chunks: list[dict] = []
    current_chunk = ""
    chunk_index = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Se o parágrafo sozinho é maior que o chunk_size,
        # divide ele em pedaços menores por sentenças
        if len(paragraph) > chunk_size:
            # Salva o chunk atual se tiver conteúdo
            if current_chunk.strip():
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "index": chunk_index,
                        "chunk_id": _generate_chunk_id(
                            source_name, chunk_index, current_chunk.strip()
                        ),
                    }
                )
                chunk_index += 1
                current_chunk = ""

            # Divide o parágrafo grande por sentenças
            sub_chunks = _split_large_paragraph(
                paragraph, chunk_size, chunk_overlap
            )
            for sc in sub_chunks:
                chunks.append(
                    {
                        "text": sc,
                        "index": chunk_index,
                        "chunk_id": _generate_chunk_id(source_name, chunk_index, sc),
                    }
                )
                chunk_index += 1
            continue

        # Verifica se adicionar este parágrafo excede o limite
        test_chunk = (
            f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
        )

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Salva o chunk atual
            if current_chunk.strip():
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "index": chunk_index,
                        "chunk_id": _generate_chunk_id(
                            source_name, chunk_index, current_chunk.strip()
                        ),
                    }
                )
                chunk_index += 1

                # Aplica sobreposição: pega o final do chunk anterior
                overlap_text = _get_overlap(current_chunk, chunk_overlap)
                current_chunk = (
                    f"{overlap_text}\n\n{paragraph}"
                    if overlap_text
                    else paragraph
                )
            else:
                current_chunk = paragraph

    # Não esquecer o último chunk
    if current_chunk.strip():
        chunks.append(
            {
                "text": current_chunk.strip(),
                "index": chunk_index,
                "chunk_id": _generate_chunk_id(
                    source_name, chunk_index, current_chunk.strip()
                ),
            }
        )

    return chunks


def _split_large_paragraph(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Divide um parágrafo grande em pedaços menores por sentenças."""
    # Tenta dividir por sentenças
    sentences = re.split(r"(?<=[.!?;])\s+", text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        test = f"{current} {sentence}" if current else sentence

        if len(test) <= chunk_size:
            current = test
        else:
            if current.strip():
                chunks.append(current.strip())
                # Sobreposição
                overlap = _get_overlap(current, chunk_overlap)
                current = f"{overlap} {sentence}" if overlap else sentence
            else:
                # Sentença individual maior que chunk_size — corta forçado
                for i in range(0, len(sentence), chunk_size - chunk_overlap):
                    chunks.append(sentence[i : i + chunk_size])
                current = ""

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _get_overlap(text: str, overlap_size: int) -> str:
    """Retorna os últimos `overlap_size` caracteres do texto."""
    if not text or overlap_size <= 0:
        return ""
    return text[-overlap_size:].strip()


def _generate_chunk_id(source_name: str, index: int, text: str) -> str:
    """Gera um ID único determinístico para o chunk."""
    content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    name_hash = hashlib.sha256(source_name.encode()).hexdigest()[:8]
    return f"chunk-{name_hash}-{index:04d}-{content_hash}"
