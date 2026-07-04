# 🛣️ RAG de Normas DNIT

Assistente de perguntas e respostas sobre as normas técnicas do DNIT (001–008), construído com **RAG (Retrieval-Augmented Generation)**. O projeto inclui um frontend **Streamlit** que expõe cada etapa do pipeline de forma didática — dos PDFs oficiais à resposta com citação de norma e seção — com os principais parâmetros ajustáveis em tempo real.

## Como funciona

```
PDFs oficiais (gov.br)
      │  normativas.py — download automatizado
      ▼
📄 Carga           PyPDFLoader: cada página vira um Document com metadados
      ▼
✂️ Chunking        RecursiveCharacterTextSplitter com regex alinhadas à
      │            estrutura das normas (seções 4.1/5.1.1, alíneas, anexos)
      ▼
🧠 Embeddings      gemini-embedding-001 → índice vetorial FAISS
      │            (cacheado em disco por parâmetros de chunking)
      ▼
🔍 Recuperação     top-k chunks por similaridade semântica
      ▼
💬 Geração         gemini-2.5-flash com grounding estrito: responde apenas
                   com base nos trechos e cita norma + seção
```

### Decisões de projeto

- **Separadores específicos do domínio** — as normas DNIT usam seções numeradas (`4.1`, `5.1.1`), alíneas (`a)`) e anexos. Cortar nessas fronteiras (via regex) faz com que ~50% dos chunks comecem exatamente em um cabeçalho de seção (contra ~7% com separadores genéricos), o que melhora a recuperação e permite ao LLM citar a seção correta.
- **Grounding estrito** — o prompt instrui o modelo a responder somente com base nos trechos recuperados e a admitir quando a informação não está na base, reduzindo alucinação.
- **Cache de índice em disco** — cada chunk custa uma chamada à API de embeddings; o índice FAISS é salvo com uma chave derivada dos parâmetros, e repetir a mesma configuração reutiliza o cache. A serialização é feita em bytes pelo Python (e não pelo `save_local` do FAISS), o que evita o bug do FAISS/C++ com caminhos acentuados no Windows.
- **Overlap onde importa** — a sobreposição entre chunks só é aplicada pelo splitter quando um corte cai no meio de uma seção; fronteiras naturais entre seções não repetem texto, evitando misturar assuntos no índice.

## Estrutura do projeto

| Arquivo | Papel |
|---|---|
| `normativas.py` | Baixa os PDFs oficiais das normas para `docs/` |
| `rag_core.py` | Núcleo do pipeline: cada etapa é uma função parametrizável |
| `app.py` | Frontend Streamlit com as etapas e parâmetros ajustáveis |
| `rag.py` | Interface de linha de comando do mesmo pipeline |
| `requirements.txt` | Dependências com versões fixadas |
| `pyproject.toml` | Metadados e configuração do linter (ruff) |

## Instalação

Pré-requisito: **Python 3.12+**

```bash
# 1. Clone o repositório
git clone https://github.com/lucasdifranco/dnit-normative-rag.git
cd dnit-normative-rag

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure a chave da API do Gemini (gratuita)
#    Obtenha em: https://aistudio.google.com/apikey
copy .env.example .env        # Windows (cp no Linux/macOS)
#    ... e edite o .env com sua chave

# 5. Baixe as normas do portal gov.br
python normativas.py
```

## Uso

### Frontend Streamlit (recomendado)

```bash
streamlit run app.py
```

O app abre em `http://localhost:8501` com quatro abas, uma por etapa do pipeline:

1. **📄 Documentos** — as normas carregadas, com páginas e tamanho de cada uma.
2. **✂️ Chunking** — métricas ao vivo (nº de chunks, tamanho médio, % que começa em cabeçalho de seção), histograma de tamanhos e amostras de chunks. Mude os parâmetros na barra lateral e veja o efeito na hora.
3. **🧠 Embeddings & Índice** — construção do índice FAISS (botão, pois consome chamadas de API; com os mesmos parâmetros o índice é reutilizado do cache em disco).
4. **💬 Consulta** — faça uma pergunta e veja **os trechos recuperados com o score de similaridade** antes da resposta final com fontes. O prompt usado fica visível em um expander.

Parâmetros ajustáveis na barra lateral:

| Parâmetro | O que controla |
|---|---|
| Tamanho do chunk | Contexto por trecho vs. precisão da busca semântica |
| Sobreposição | Texto repetido entre chunks vizinhos em cortes no meio de seção |
| Separadores | Estrutura DNIT (seções/alíneas) vs. genéricos (parágrafos) |
| k (recuperação) | Quantos trechos são enviados ao LLM |
| Modelo / Temperature | Modelo Gemini usado na geração e seu determinismo |

### Linha de comando

```bash
python rag.py                                  # perguntas de demonstração
python rag.py "O que é uma norma PRO?"         # pergunta específica
python rag.py -k 6 "Como numerar as seções?"   # ajustando a recuperação
```

## Desenvolvimento

```bash
pip install ruff
ruff check .      # lint
ruff format .     # formatação
```

## Stack

Python 3.12 · LangChain 1.x · Google Gemini (embeddings + geração) · FAISS · Streamlit

## Licença

[MIT](LICENSE)
