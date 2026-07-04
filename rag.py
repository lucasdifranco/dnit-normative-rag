"""
Versão de linha de comando do pipeline RAG.

Usa as mesmas funções do rag_core (e o mesmo cache de índice em disco)
que o frontend Streamlit.

Uso:
    python rag.py                        # roda as perguntas de demonstração
    python rag.py "Sua pergunta aqui"    # responde uma pergunta específica
    python rag.py -k 6 "Sua pergunta"    # ajusta quantos chunks são recuperados
"""

import argparse
from pathlib import Path

import rag_core

PERGUNTAS_DEMO = [
    "Como deve ser estruturada uma norma técnica do DNIT?",
    "O que é uma norma do tipo Procedimento (PRO)?",
    "Quais são os elementos preliminares de uma norma?",
]


def responder(indice, pergunta: str, k: int) -> None:
    print(f"\n{'=' * 60}\nPERGUNTA: {pergunta}\n")
    resultados = rag_core.recuperar(indice, pergunta, k=k)
    resposta = rag_core.gerar_resposta(pergunta, [doc for doc, _ in resultados])
    print(f"RESPOSTA: {resposta}\n")
    print("FONTES (score = distância L2; menor é mais similar):")
    for doc, score in resultados:
        origem = Path(doc.metadata.get("source", "?")).name
        print(f"  - {score:.3f} | {origem} | página {doc.metadata.get('page', '?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG de normas DNIT (CLI)")
    parser.add_argument("pergunta", nargs="?", help="pergunta a responder (opcional)")
    parser.add_argument("-k", type=int, default=4, help="chunks recuperados (padrão: 4)")
    args = parser.parse_args()

    print("Carregando documentos...")
    docs = rag_core.carregar_documentos()
    chunks = rag_core.dividir_chunks(docs)
    print(f"{len(docs)} páginas -> {len(chunks)} chunks")

    print("Construindo/carregando índice (cache em disco)...")
    indice = rag_core.construir_indice(chunks)
    print(f"Índice pronto: {indice.index.ntotal} vetores.")

    perguntas = [args.pergunta] if args.pergunta else PERGUNTAS_DEMO
    for pergunta in perguntas:
        responder(indice, pergunta, k=args.k)


if __name__ == "__main__":
    main()
