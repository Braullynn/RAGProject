"""
Pipeline de ingestão de dados multimodais.
Orquestra a leitura de arquivos, geração de embeddings e armazenamento no Pinecone.
"""

from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from .chunking import _generate_chunk_id, chunk_text
from .config import EXTENSION_TO_MIME, Settings
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore

console = Console()


class IngestPipeline:
    """Pipeline para ingestão de arquivos no banco vetorial."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.engine = EmbeddingEngine(self.settings)
        self.store = VectorStore(self.settings)

    def ingest_file(self, file_path: str | Path, namespace: str = "") -> int:
        """
        Ingere um único arquivo: gera embedding e armazena no Pinecone.

        Returns:
            Número de vetores inseridos (geralmente 1, pode ser > 1 para vídeos).
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Caminho não é um arquivo: {file_path}")

        console.print(f"\n📂 Processando: [bold]{file_path.name}[/bold]")
        
        # Otimização: Coletar possíveis IDs primeiro para evitar reprocessamento
        existing_ids = set()
        mime_type = self.engine._get_mime_type(file_path)
        
        if mime_type.startswith("text/"):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_text(text, source_name=file_path.name)
            possible_ids = [c["chunk_id"] for c in chunks]
            existing_ids = self.store.check_existing_ids(possible_ids, namespace=namespace)
        elif mime_type == "application/pdf":
            import fitz
            doc = fitz.open(file_path)
            extracted_text = ""
            for i, page in enumerate(doc):
                extracted_text += page.get_text("text") + "\n"
            doc.close()
            chunks = chunk_text(extracted_text, source_name=file_path.name)
            possible_ids = [c["chunk_id"] for c in chunks]
            existing_ids = self.store.check_existing_ids(possible_ids, namespace=namespace)
            
        if existing_ids:
            console.print(f"  🔍 Encontrados {len(existing_ids)} chunk(s) já ingeridos no banco.")

        # Gera embedding(s) pulando os que já existem
        vectors = self.engine.embed_content(file_path, existing_ids=existing_ids)
        
        if not vectors:
            console.print("  ✅ Nenhum vetor novo para inserir.", style="green")
            return 0

        # Adiciona metadados de ingestão
        timestamp = datetime.now(timezone.utc).isoformat()
        for vec in vectors:
            vec["metadata"]["source_path"] = str(file_path.resolve())
            vec["metadata"]["ingested_at"] = timestamp

        # Armazena no Pinecone
        count = self.store.upsert_vectors(vectors, namespace=namespace)
        console.print(
            f"  ✅ {count} vetor(es) armazenado(s) no Pinecone.",
            style="green",
        )

        return count

    def ingest_directory(
        self,
        dir_path: str | Path,
        namespace: str = "",
        recursive: bool = True,
    ) -> dict:
        """
        Ingere todos os arquivos suportados de um diretório.

        Returns:
            Dicionário com estatísticas da ingestão.
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Caminho não é um diretório: {dir_path}")

        # Coleta arquivos suportados
        supported_extensions = set(EXTENSION_TO_MIME.keys())
        files: list[Path] = []

        pattern = "**/*" if recursive else "*"
        for p in dir_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in supported_extensions:
                files.append(p)

        if not files:
            console.print(
                f"  ⚠️  Nenhum arquivo suportado encontrado em: {dir_path}",
                style="yellow",
            )
            return {"total_files": 0, "total_vectors": 0, "errors": []}

        console.print(
            f"\n{'='*60}\n"
            f"📁 Ingestão em lote: [bold]{dir_path}[/bold]\n"
            f"   {len(files)} arquivo(s) encontrado(s)\n"
            f"{'='*60}"
        )

        stats = {
            "total_files": len(files),
            "processed": 0,
            "total_vectors": 0,
            "errors": [],
            "by_type": {},
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Ingerindo...", total=len(files))

            for file_path in files:
                try:
                    count = self.ingest_file(file_path, namespace=namespace)
                    stats["processed"] += 1
                    stats["total_vectors"] += count

                    # Contagem por tipo
                    ext = file_path.suffix.lower()
                    stats["by_type"][ext] = stats["by_type"].get(ext, 0) + 1

                except Exception as e:
                    error_msg = f"{file_path.name}: {e}"
                    stats["errors"].append(error_msg)
                    console.print(f"  ❌ Erro: {error_msg}", style="red")

                progress.advance(task)

        # Resumo final
        console.print(
            f"\n{'='*60}\n"
            f"📊 Resumo da Ingestão:\n"
            f"   Arquivos processados: {stats['processed']}/{stats['total_files']}\n"
            f"   Vetores armazenados:  {stats['total_vectors']}\n"
            f"   Erros:               {len(stats['errors'])}\n"
            f"   Por tipo:            {stats['by_type']}\n"
            f"{'='*60}"
        )

        return stats

    def ingest_text(self, text: str, namespace: str = "") -> int:
        """
        Ingere texto diretamente (sem arquivo).
        Útil para testes ou texto vindo de APIs.
        """
        console.print(f"\n📝 Ingerindo texto ({len(text)} caracteres)...")

        vector = self.engine.embed_text(text)

        # Metadados de ingestão
        vector["metadata"]["source_path"] = "inline_text"
        vector["metadata"]["ingested_at"] = datetime.now(timezone.utc).isoformat()

        count = self.store.upsert_vectors([vector], namespace=namespace)
        console.print(
            f"  ✅ {count} vetor armazenado no Pinecone.", style="green"
        )

        return count
