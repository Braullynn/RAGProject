"""
Motor de busca semântica multimodal.
Permite buscar conteúdo por texto, imagem ou qualquer arquivo suportado.
"""

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import Settings
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore

console = Console()


class SearchEngine:
    """Motor de busca semântica no banco vetorial."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.engine = EmbeddingEngine(self.settings)
        self.store = VectorStore(self.settings)

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "",
        filter_type: str | None = None,
    ) -> list[dict]:
        """
        Busca semântica por texto.

        Args:
            query: Texto da consulta.
            top_k: Número de resultados.
            namespace: Namespace para buscar.
            filter_type: Filtrar por tipo (texto, imagem, video, audio, documento).

        Returns:
            Lista de resultados com score e metadados.
        """
        console.print(f'\n🔍 Buscando: "[bold]{query}[/bold]"')

        # Gera embedding da query
        query_vector = self.engine.embed_query(query)

        # Monta filtro
        filter_dict = None
        if filter_type:
            filter_dict = {"type": {"$eq": filter_type}}

        # Busca no Pinecone
        results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict,
        )

        self._print_results(results)
        return results

    def search_by_image(
        self,
        image_path: str | Path,
        top_k: int = 5,
        namespace: str = "",
        filter_type: str | None = None,
    ) -> list[dict]:
        """Busca conteúdo similar a uma imagem."""
        image_path = Path(image_path)
        console.print(f"\n🔍 Buscando similar a: [bold]{image_path.name}[/bold]")

        # Gera embedding da imagem
        result = self.engine.embed_image(image_path)
        query_vector = result["values"]

        # Monta filtro
        filter_dict = None
        if filter_type:
            filter_dict = {"type": {"$eq": filter_type}}

        # Busca no Pinecone
        results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict,
        )

        self._print_results(results)
        return results

    def search_similar(
        self,
        file_path: str | Path,
        top_k: int = 5,
        namespace: str = "",
        filter_type: str | None = None,
    ) -> list[dict]:
        """
        Busca conteúdo similar a qualquer arquivo suportado.
        Detecta o tipo automaticamente.
        """
        file_path = Path(file_path)
        console.print(f"\n🔍 Buscando similar a: [bold]{file_path.name}[/bold]")

        # Gera embedding do arquivo
        vectors = self.engine.embed_content(file_path)
        query_vector = vectors[0]["values"]

        # Monta filtro
        filter_dict = None
        if filter_type:
            filter_dict = {"type": {"$eq": filter_type}}

        # Busca no Pinecone
        results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict,
        )

        self._print_results(results)
        return results

    @staticmethod
    def _print_results(results: list[dict]) -> None:
        """Imprime os resultados de forma bonita no terminal."""
        if not results:
            console.print("  😕 Nenhum resultado encontrado.", style="yellow")
            return

        table = Table(
            title=f"🎯 {len(results)} resultado(s) encontrado(s)",
            show_lines=True,
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Tipo", style="magenta", width=12)
        table.add_column("Arquivo / Conteúdo", style="green")

        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            content_type = meta.get("type", "?")

            # Determina o que mostrar como identificador
            if "file_name" in meta:
                identifier = meta["file_name"]
            elif "content_preview" in meta:
                preview = meta["content_preview"]
                identifier = (
                    f"{preview[:80]}..." if len(preview) > 80 else preview
                )
            else:
                identifier = result["id"]

            score = f"{result['score']:.4f}"
            table.add_row(str(i), score, content_type, identifier)

        console.print(table)
