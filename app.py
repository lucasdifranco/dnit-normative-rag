"""
Frontend Streamlit do RAG de normas DNIT.

Mostra o pipeline etapa por etapa (documentos -> chunking -> índice ->
recuperação -> geração), com os parâmetros de cada fase ajustáveis na
barra lateral. Rode com:

    streamlit run app.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import rag_core

st.set_page_config(page_title="RAG Normas DNIT", page_icon="🛣️", layout="wide")

st.title("🛣️ RAG de Normas DNIT")
st.markdown(
    "Assistente de perguntas e respostas sobre as normas DNIT 001–008, construído com "
    "**RAG (Retrieval-Augmented Generation)**: os PDFs oficiais são divididos em trechos, "
    "indexados por similaridade semântica e usados como única fonte para as respostas do LLM. "
    "Cada aba abaixo mostra uma etapa do pipeline — ajuste os parâmetros na barra lateral "
    "e veja o efeito em cada fase."
)

# ----------------------------------------------------------------- SIDEBAR
st.sidebar.header("⚙️ Parâmetros do pipeline")

st.sidebar.subheader("✂️ Chunking")
chunk_size = st.sidebar.slider(
    "Tamanho do chunk (caracteres)",
    200,
    2000,
    800,
    step=100,
    help="Chunks maiores dão mais contexto ao LLM, mas diluem a similaridade "
    "semântica e podem misturar assuntos diferentes no mesmo trecho.",
)
chunk_overlap = st.sidebar.slider(
    "Sobreposição (caracteres)",
    0,
    400,
    100,
    step=50,
    help="Trecho repetido entre chunks vizinhos, para não perder informação "
    "que caia exatamente na fronteira do corte.",
)
estrategia = st.sidebar.radio(
    "Separadores",
    ["Estrutura DNIT (seções/alíneas)", "Genéricos (parágrafos)"],
    help="Os separadores DNIT cortam o texto nas fronteiras de seções numeradas "
    "(4.1, 5.1.1...), alíneas e anexos — cada chunk tende a ser uma seção completa.",
)
separadores = (
    rag_core.SEPARADORES_DNIT
    if estrategia.startswith("Estrutura")
    else rag_core.SEPARADORES_GENERICOS
)

st.sidebar.subheader("🔍 Recuperação")
k = st.sidebar.slider(
    "Chunks recuperados (k)",
    1,
    10,
    4,
    help="Quantos trechos mais similares à pergunta são enviados ao LLM. "
    "Mais trechos = mais contexto, porém mais ruído e mais tokens.",
)

st.sidebar.subheader("💬 Geração")
modelo_llm = st.sidebar.selectbox(
    "Modelo LLM",
    ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"],
)
temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    1.0,
    0.0,
    step=0.1,
    help="0 = respostas determinísticas e fiéis ao texto (recomendado para "
    "consulta normativa); valores altos aumentam a criatividade e o risco de alucinação.",
)


# ------------------------------------------------------- DADOS (com cache)
@st.cache_data(show_spinner="Carregando PDFs...")
def carregar_docs():
    return rag_core.carregar_documentos()


@st.cache_data(show_spinner="Dividindo em chunks...")
def dividir(chunk_size: int, chunk_overlap: int, separadores: tuple):
    return rag_core.dividir_chunks(carregar_docs(), chunk_size, chunk_overlap, list(separadores))


docs = carregar_docs()
chunks = dividir(chunk_size, chunk_overlap, tuple(separadores))

aba_docs, aba_chunks, aba_indice, aba_consulta = st.tabs(
    ["📄 1. Documentos", "✂️ 2. Chunking", "🧠 3. Embeddings & Índice", "💬 4. Consulta"]
)

# ------------------------------------------------------------ 1. DOCUMENTOS
with aba_docs:
    st.markdown(
        "#### Carga dos documentos\n"
        "Os PDFs oficiais das normas (baixados do portal gov.br pelo `normativas.py`) "
        "são lidos com o `PyPDFLoader` — cada página vira um `Document` com metadados "
        "de origem, que acompanham o texto até a citação da fonte na resposta final."
    )
    tabela = {}
    for d in docs:
        nome = Path(d.metadata.get("source", "?")).name
        info = tabela.setdefault(nome, {"páginas": 0, "caracteres": 0})
        info["páginas"] += 1
        info["caracteres"] += len(d.page_content)
    df_docs = pd.DataFrame([{"norma": n, **i} for n, i in sorted(tabela.items())])
    col1, col2 = st.columns([2, 1])
    col1.dataframe(df_docs, width="stretch", hide_index=True)
    col2.metric("Normas carregadas", len(tabela))
    col2.metric("Total de páginas", len(docs))

# -------------------------------------------------------------- 2. CHUNKING
with aba_chunks:
    st.markdown(
        "#### Divisão em chunks\n"
        "O texto é dividido pelo `RecursiveCharacterTextSplitter`, que tenta cortar "
        "primeiro nos separadores de nível mais alto. Como as normas DNIT usam seções "
        "numeradas (`4.1`, `5.1.1`), alíneas (`a)`) e anexos, os separadores foram "
        "definidos por **regex alinhadas a essa estrutura** — assim cada chunk tende a "
        "conter uma seção completa, com o número que o LLM usa para citar a fonte."
    )
    tamanhos = [len(c.page_content) for c in chunks]
    inicia_secao = sum(
        1
        for c in chunks
        if re.match(r"^(\d+(\.\d+)*\s|Anexo\s[A-Z]|[a-z]\)\s)", c.page_content.strip())
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chunks gerados", len(chunks))
    m2.metric("Tamanho médio", f"{sum(tamanhos) // max(len(tamanhos), 1)} chars")
    m3.metric("Maior chunk", f"{max(tamanhos, default=0)} chars")
    m4.metric(
        "Começam em seção",
        f"{100 * inicia_secao // max(len(chunks), 1)}%",
        help="Percentual de chunks cujo início coincide com um cabeçalho de "
        "seção/alínea — indicador da qualidade das fronteiras do corte.",
    )

    hist = pd.cut(pd.Series(tamanhos), bins=10).value_counts().sort_index()
    hist.index = [f"{int(i.left)}–{int(i.right)}" for i in hist.index]
    st.markdown("**Distribuição dos tamanhos de chunk (caracteres)**")
    st.bar_chart(hist)

    st.markdown("**Amostra de chunks** — observe onde cada um começa:")
    for i, c in enumerate(chunks[10:16]):
        origem = Path(c.metadata.get("source", "?")).name
        with st.expander(
            f"Chunk {i + 10} · {origem} · pág. {c.metadata.get('page', '?')} · "
            f"{len(c.page_content)} chars — “{c.page_content.strip()[:60]}...”"
        ):
            st.text(c.page_content)

# ----------------------------------------------------- 3. EMBEDDINGS/ÍNDICE
with aba_indice:
    st.markdown(
        "#### Embeddings e índice vetorial\n"
        "Cada chunk é convertido em um vetor pelo modelo `gemini-embedding-001` e "
        "armazenado em um índice **FAISS**, que permite buscar os trechos mais "
        "próximos de uma pergunta por similaridade semântica.\n\n"
        "⚠️ Cada chunk custa **uma chamada à API de embeddings**, por isso a "
        "construção é manual (botão abaixo) e o índice é **cacheado em disco**: "
        "repetir os mesmos parâmetros de chunking reutiliza o índice já construído."
    )
    if st.button("🧱 Construir / carregar índice", type="primary"):
        inicio = time.time()
        with st.spinner(f"Indexando {len(chunks)} chunks..."):
            st.session_state.indice = rag_core.construir_indice(chunks)
            st.session_state.params_indice = (chunk_size, chunk_overlap, estrategia)
        st.success(f"Índice pronto em {time.time() - inicio:.1f}s.")

    if "indice" in st.session_state:
        indice = st.session_state.indice
        c1, c2, c3 = st.columns(3)
        c1.metric("Vetores no índice", indice.index.ntotal)
        c2.metric("Dimensão do vetor", indice.index.d)
        c3.metric(
            "Chunking usado",
            f"{st.session_state.params_indice[0]}/{st.session_state.params_indice[1]}",
            help="chunk_size / overlap com que o índice foi construído.",
        )
        if st.session_state.params_indice != (chunk_size, chunk_overlap, estrategia):
            st.warning(
                "Os parâmetros de chunking mudaram desde a construção do índice. "
                "Reconstrua-o para que a busca reflita os chunks atuais."
            )
    else:
        st.info("Nenhum índice em memória ainda — clique no botão acima.")

# --------------------------------------------------------------- 4. CONSULTA
with aba_consulta:
    st.markdown(
        "#### Recuperação + geração com grounding estrito\n"
        "A pergunta é vetorizada com o mesmo modelo de embeddings e o índice devolve "
        f"os **{k} chunks mais próximos**, que são inseridos em um prompt que instrui o "
        "LLM a responder **somente com base neles** e a citar norma e seção — se a "
        "informação não estiver nos trechos, o assistente diz que não encontrou."
    )
    with st.expander("📝 Ver o prompt usado na geração"):
        st.code(rag_core.PROMPT_RAG.template, language="text")

    exemplos = [
        "Como deve ser estruturada uma norma técnica do DNIT?",
        "O que é uma norma do tipo Procedimento (PRO)?",
        "Quais são os elementos preliminares de uma norma?",
    ]
    cols = st.columns(len(exemplos))
    for col, ex in zip(cols, exemplos, strict=True):
        if col.button(ex, width="stretch"):
            st.session_state.pergunta = ex

    pergunta = st.text_input(
        "Pergunta sobre as normas:",
        value=st.session_state.get("pergunta", ""),
        placeholder="Ex.: Como é feita a numeração das seções de uma norma?",
    )

    if st.button("🚀 Perguntar", type="primary", disabled=not pergunta):
        if "indice" not in st.session_state:
            st.error("Construa o índice primeiro (aba 3).")
        else:
            with st.spinner("Buscando trechos relevantes..."):
                resultados = rag_core.recuperar(st.session_state.indice, pergunta, k=k)

            st.markdown("##### 🔍 Trechos recuperados")
            st.caption("Score = distância L2 entre os vetores (menor = mais similar à pergunta).")
            for pos, (doc, score) in enumerate(resultados, start=1):
                origem = Path(doc.metadata.get("source", "?")).name
                with st.expander(
                    f"#{pos} · score {score:.3f} · {origem} · pág. {doc.metadata.get('page', '?')}"
                ):
                    st.text(doc.page_content)

            with st.spinner(f"Gerando resposta com {modelo_llm}..."):
                resposta = rag_core.gerar_resposta(
                    pergunta,
                    [doc for doc, _ in resultados],
                    modelo_llm=modelo_llm,
                    temperature=temperature,
                )
            st.markdown("##### 💬 Resposta")
            st.markdown(resposta)
