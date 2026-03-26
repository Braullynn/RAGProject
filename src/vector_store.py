"""
Integração com o Pinecone Vector Database.
Gerencia criação de índice, upsert e busca de vetores.
"""

import time

from pinecone import Pinecone, ServerlessSpec
from rich.console import Console

from .config import Settings

console = Console()


class VectorStore:
    """Gerencia o índice vetorial no Pinecone."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self._index = None

    @property
    def index(self):
        """Retorna a referência ao índice, conectando se necessário."""
        if self._index is None:
            self._index = self.pc.Index(self.settings.pinecone_index_name)
        return self._index

    # ------------------------------------------------------------------
    # Gerenciamento do Índice
    # ------------------------------------------------------------------

    def init_index(self) -> None:
        """
        Cria o índice serverless no Pinecone se não existir.
        Se já existir, apenas conecta.
        """
        index_name = self.settings.pinecone_index_name
        existing = [idx.name for idx in self.pc.list_indexes()]

        if index_name in existing:
            console.print(
                f"  ✅ Índice '{index_name}' já existe. Conectando...",
                style="green",
            )
        else:
            console.print(
                f"  🔨 Criando índice '{index_name}' "
                f"({self.settings.embedding_dimensions}D, cosine)...",
                style="yellow",
            )
            self.pc.create_index(
                name=index_name,
                dimension=self.settings.embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
            # Aguarda o índice ficar pronto
            console.print("  ⏳ Aguardando índice ficar pronto...")
            while not self.pc.describe_index(index_name).status.get("ready"):
                time.sleep(1)
            console.print(
                f"  ✅ Índice '{index_name}' criado com sucesso!", style="green"
            )

        # Conecta ao índice
        self._index = self.pc.Index(index_name)

    def get_stats(self) -> dict:
        """Retorna estatísticas do índice."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": {
                ns: data.vector_count
                for ns, data in (stats.namespaces or {}).items()
            },
        }

    # ------------------------------------------------------------------
    # Operações com Vetores
    # ------------------------------------------------------------------

    def upsert_vectors(
        self,
        vectors: list[dict],
        namespace: str = "",
        batch_size: int = 50,
    ) -> int:
        """
        Insere/atualiza vetores no Pinecone.

        Args:
            vectors: Lista de dicts com 'id', 'values' e 'metadata'.
            namespace: Namespace opcional para organizar os vetores.
            batch_size: Tamanho do lote para upsert.

        Returns:
            Número de vetores inseridos com sucesso.
        """
        total = 0

        # Processa em lotes
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]

            # Formata para o Pinecone
            pinecone_vectors = []
            for vec in batch:
                pinecone_vectors.append(
                    {
                        "id": vec["id"],
                        "values": vec["values"],
                        "metadata": vec.get("metadata", {}),
                    }
                )

            self.index.upsert(vectors=pinecone_vectors, namespace=namespace)
            total += len(batch)

        return total

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        namespace: str = "",
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """
        Busca por similaridade no Pinecone.

        Args:
            query_vector: Vetor de embedding da query.
            top_k: Número de resultados.
            namespace: Namespace para buscar.
            filter_dict: Filtros de metadados (ex: {"type": "imagem"}).

        Returns:
            Lista de resultados com score e metadados.
        """
        kwargs = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": namespace,
        }
        if filter_dict:
            kwargs["filter"] = filter_dict

        results = self.index.query(**kwargs)

        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata or {},
            }
            for match in results.matches
        ]

    def delete_vectors(
        self,
        ids: list[str],
        namespace: str = "",
    ) -> None:
        """Remove vetores por ID."""
        self.index.delete(ids=ids, namespace=namespace)
        console.print(
            f"  🗑️  {len(ids)} vetores removidos.", style="red"
        )

    def check_existing_ids(
        self,
        ids: list[str],
        namespace: str = "",
        batch_size: int = 200,
    ) -> set[str]:
        """
        Verifica quais dos IDs fornecidos já existem no índice.
        Retorna um set com os IDs que JÁ EXISTEM.
        """
        if not ids:
            return set()
            
        existing_ids = set()
        
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            try:
                response = self.index.fetch(ids=batch, namespace=namespace)
                for vector_id in response.vectors.keys():
                    existing_ids.add(vector_id)
            except Exception as e:
                console.print(f"  ⚠️ Erro ao verificar batch no Pinecone: {e}", style="yellow")
                
        return existing_ids

    def delete_all(self, namespace: str = "") -> None:
        """Remove todos os vetores do namespace."""
        self.index.delete(delete_all=True, namespace=namespace)
        console.print("  🗑️  Todos os vetores removidos.", style="red")
