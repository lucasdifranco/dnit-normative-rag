"""
Núcleo do pipeline RAG para as normas DNIT.

Cada etapa do pipeline é uma função parametrizável, para que o frontend
(app.py) possa expor os parâmetros e mostrar o resultado de cada fase:

    1. carregar_documentos()  -> páginas dos PDFs
    2. dividir_chunks()       -> chunks com fronteiras nas seções das normas
    3. construir_indice()     -> embeddings + índice FAISS (com cache em disco)
    4. recuperar()            -> top-k chunks mais similares à pergunta
    5. gerar_resposta()       -> resposta do LLM com grounding estrito
"""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

RAIZ = Path(__file__).parent
PASTA_DOCS = RAIZ / "docs"
PASTA_INDICES = RAIZ / "indices"

# Separadores alinhados à estrutura das normas DNIT (seções numeradas,
# alíneas e anexos), do nível mais alto para o mais baixo.
SEPARADORES_DNIT = [
    r"\nAnexo [A-Z]",  # anexos: "Anexo A – ..."
    r"\n\d+ [A-ZÀ-Ú]",  # seções: "4 Estrutura de uma norma técnica"
    r"\n\d+\.\d+ ",  # subseções: "4.1 Disposição dos elementos"
    r"\n\d+\.\d+\.\d+ ",  # subsubseções: "5.1.1 ..."
    r"\n[a-z]\) ",  # alíneas: "a) ..."
    "\n\n",
    "\n",
    " ",
    "",
]

# Separadores genéricos, para comparação didática no frontend.
SEPARADORES_GENERICOS = ["\n\n", "\n", " ", ""]

PROMPT_RAG = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Você é um assistente técnico especializado em normas rodoviárias do DNIT.
Responda SOMENTE com base nos trechos abaixo.
Se a informação não estiver nos trechos, diga explicitamente:
"Não encontrei essa informação na base."
Para cada afirmação, cite a fonte (norma e seção/subseção se disponível,
ex.: DNIT 001/2023-PRO, seção 4.1).

TRECHOS:
{context}

PERGUNTA: {question}

RESPOSTA:""",
)


def _api_key() -> str:
    chave = os.getenv("GEMINI_API_KEY")
    if not chave:
        raise RuntimeError("GEMINI_API_KEY não encontrada. Defina-a no arquivo .env.")
    return chave


# ---------------------------------------------------------------- 1. CARGA
def carregar_documentos(pasta: Path = PASTA_DOCS) -> list[Document]:
    """Carrega todos os PDFs da pasta; cada página vira um Document."""
    docs = []
    for arquivo in sorted(pasta.glob("*.pdf")):
        docs.extend(PyPDFLoader(str(arquivo)).load())
    return docs


# ------------------------------------------------------------- 2. CHUNKING
def dividir_chunks(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separadores: list[str] | None = None,
) -> list[Document]:
    """Divide as páginas em chunks respeitando a hierarquia das normas."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separadores or SEPARADORES_DNIT,
        is_separator_regex=True,
    )
    return splitter.split_documents(docs)


# -------------------------------------------------------------- 3. ÍNDICE
def _id_indice(chunks: list[Document], modelo_embedding: str) -> str:
    """Identificador determinístico do índice em função do conteúdo e parâmetros."""
    h = hashlib.sha256()
    h.update(modelo_embedding.encode())
    h.update(str(len(chunks)).encode())
    for c in chunks[:: max(1, len(chunks) // 50)]:  # amostra estável do conteúdo
        h.update(c.page_content[:200].encode())
    return h.hexdigest()[:16]


def construir_indice(
    chunks: list[Document],
    modelo_embedding: str = "models/gemini-embedding-001",
    usar_cache: bool = True,
) -> FAISS:
    """
    Gera embeddings e monta o índice FAISS.

    Como cada chunk vira uma chamada à API de embeddings, o índice é salvo em
    disco com uma chave derivada dos parâmetros — mudar chunk_size/overlap ou
    o modelo gera um índice novo; repetir os mesmos parâmetros reutiliza o cache.

    A (de)serialização é feita em bytes pelo Python (serialize_to_bytes) em vez
    de save_local/load_local: o FAISS em C++ não abre caminhos com caracteres
    acentuados no Windows (ex.: "Portifólio").
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=modelo_embedding, google_api_key=_api_key())
    caminho = PASTA_INDICES / f"{_id_indice(chunks, modelo_embedding)}.faiss"
    if usar_cache and caminho.exists():
        return FAISS.deserialize_from_bytes(
            caminho.read_bytes(), embeddings, allow_dangerous_deserialization=True
        )
    indice = FAISS.from_documents(chunks, embeddings)
    if usar_cache:
        PASTA_INDICES.mkdir(exist_ok=True)
        caminho.write_bytes(indice.serialize_to_bytes())
    return indice


# ---------------------------------------------------------- 4. RECUPERAÇÃO
def recuperar(indice: FAISS, pergunta: str, k: int = 4) -> list[tuple[Document, float]]:
    """Retorna os k chunks mais próximos da pergunta, com a distância L2 de cada um."""
    return indice.similarity_search_with_score(pergunta, k=k)


# -------------------------------------------------------------- 5. GERAÇÃO
def gerar_resposta(
    pergunta: str,
    trechos: list[Document],
    modelo_llm: str = "gemini-2.5-flash",
    temperature: float = 0.0,
) -> str:
    """Monta o prompt com os trechos recuperados e chama o LLM."""
    llm = ChatGoogleGenerativeAI(
        model=modelo_llm, temperature=temperature, google_api_key=_api_key()
    )
    contexto = "\n\n---\n\n".join(
        f"[{Path(d.metadata.get('source', '?')).name} | página {d.metadata.get('page', '?')}]\n"
        f"{d.page_content}"
        for d in trechos
    )
    mensagem = PROMPT_RAG.format(context=contexto, question=pergunta)
    return llm.invoke(mensagem).content
