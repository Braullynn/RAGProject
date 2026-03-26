"""
CLI Principal do RAG Multimodal — Gemini Embeddings 2 + Pinecone.

Uso:
    python main.py init                         Inicializar índice no Pinecone
    python main.py ingest <caminho>             Ingerir arquivo ou diretório
    python main.py ingest --text "conteúdo"     Ingerir texto diretamente
    python main.py search "<query>"             Buscar por texto
    python main.py search --image <caminho>     Buscar por imagem similar
    python main.py search --file <caminho>      Buscar por arquivo similar
    python main.py stats                        Estatísticas do índice
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from src.config import Settings
from src.ingest import IngestPipeline
from src.search import SearchEngine
from src.vector_store import VectorStore

console = Console()


def print_banner() -> None:
    """Imprime o banner do projeto."""
    banner = (
        "[bold cyan]🧠 RAG Multimodal[/bold cyan]\n"
        "[dim]Gemini Embedding 2 + Pinecone[/dim]"
    )
    console.print(Panel(banner, expand=False, border_style="cyan"))


def cmd_init(settings: Settings) -> None:
    """Inicializa o índice no Pinecone."""
    console.print("\n🔧 [bold]Inicializando índice no Pinecone...[/bold]\n")
    store = VectorStore(settings)
    store.init_index()
    console.print("\n✅ Índice pronto para uso!\n", style="bold green")


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> None:
    """Ingere arquivos ou texto no banco vetorial."""
    pipeline = IngestPipeline(settings)

    if args.text:
        # Ingestão de texto direto
        pipeline.ingest_text(args.text, namespace=args.namespace)
    elif args.path:
        from pathlib import Path

        path = Path(args.path)

        if path.is_file():
            pipeline.ingest_file(path, namespace=args.namespace)
        elif path.is_dir():
            pipeline.ingest_directory(
                path,
                namespace=args.namespace,
                recursive=not args.no_recursive,
            )
        else:
            console.print(
                f"❌ Caminho não encontrado: {args.path}", style="red"
            )
            sys.exit(1)
    else:
        console.print(
            "❌ Informe um caminho ou use --text para ingerir texto.",
            style="red",
        )
        sys.exit(1)


def cmd_search(args: argparse.Namespace, settings: Settings) -> None:
    """Realiza busca semântica."""
    search = SearchEngine(settings)

    if args.image:
        search.search_by_image(
            args.image,
            top_k=args.top_k,
            namespace=args.namespace,
            filter_type=args.filter,
        )
    elif args.file:
        search.search_similar(
            args.file,
            top_k=args.top_k,
            namespace=args.namespace,
            filter_type=args.filter,
        )
    elif args.query:
        search.search_by_text(
            args.query,
            top_k=args.top_k,
            namespace=args.namespace,
            filter_type=args.filter,
        )
    else:
        console.print(
            "❌ Informe uma query, --image ou --file para buscar.",
            style="red",
        )
        sys.exit(1)


def cmd_stats(settings: Settings) -> None:
    """Mostra estatísticas do índice."""
    store = VectorStore(settings)
    stats = store.get_stats()

    console.print("\n📊 [bold]Estatísticas do Índice[/bold]\n")
    console.print(f"   Vetores totais: [cyan]{stats['total_vectors']}[/cyan]")
    console.print(f"   Dimensão:       [cyan]{stats['dimension']}[/cyan]")

    if stats["namespaces"]:
        console.print("   Namespaces:")
        for ns, count in stats["namespaces"].items():
            ns_label = ns if ns else "(default)"
            console.print(f"     • {ns_label}: {count} vetores")
    console.print()


def main() -> None:
    """Ponto de entrada principal da CLI."""
    parser = argparse.ArgumentParser(
        description="RAG Multimodal — Gemini Embedding 2 + Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")

    # --- init ---
    subparsers.add_parser("init", help="Inicializar índice no Pinecone")

    # --- ingest ---
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingerir arquivo, diretório ou texto"
    )
    ingest_parser.add_argument(
        "path", nargs="?", help="Caminho do arquivo ou diretório"
    )
    ingest_parser.add_argument(
        "--text", "-t", help="Texto para ingerir diretamente"
    )
    ingest_parser.add_argument(
        "--namespace", "-n", default="", help="Namespace do Pinecone"
    )
    ingest_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Não buscar em subdiretórios",
    )

    # --- search ---
    search_parser = subparsers.add_parser(
        "search", help="Busca semântica"
    )
    search_parser.add_argument(
        "query", nargs="?", help="Texto da consulta"
    )
    search_parser.add_argument(
        "--image", "-i", help="Buscar por imagem similar"
    )
    search_parser.add_argument(
        "--file", "-f", help="Buscar por arquivo similar"
    )
    search_parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Número de resultados (padrão: 5)"
    )
    search_parser.add_argument(
        "--namespace", "-n", default="", help="Namespace do Pinecone"
    )
    search_parser.add_argument(
        "--filter",
        choices=["texto", "imagem", "video", "audio", "documento"],
        help="Filtrar por tipo de conteúdo",
    )

    # --- stats ---
    subparsers.add_parser("stats", help="Estatísticas do índice")

    # Parse
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Banner
    print_banner()

    # Carrega e valida configurações
    settings = Settings()
    console.print(f"\n{settings}\n")

    if not settings.validate():
        console.print(
            "\n❌ Configure suas chaves de API no arquivo .env\n",
            style="bold red",
        )
        sys.exit(1)

    # Executa o comando
    if args.command == "init":
        cmd_init(settings)
    elif args.command == "ingest":
        cmd_ingest(args, settings)
    elif args.command == "search":
        cmd_search(args, settings)
    elif args.command == "stats":
        cmd_stats(settings)


if __name__ == "__main__":
    main()
