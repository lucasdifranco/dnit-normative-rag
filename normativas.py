"""
Baixa as normas DNIT 001-008 (vigentes) do portal oficial gov.br
e gera um arquivo normas_dnit_001_008.zip.
"""

import zipfile
from pathlib import Path

import requests

BASE = (
    "https://www.gov.br/dnit/pt-br/assuntos/planejamento-e-pesquisa/"
    "ipr/coletanea-de-normas/coletanea-de-normas"
)

NORMAS = {
    "DNIT_001_2023_PRO.pdf": f"{BASE}/procedimento-pro/dnit_001_2023_pro.pdf",
    "DNIT_002_2023_PRO.pdf": f"{BASE}/procedimento-pro/dnit_002_2023_pro.pdf",
    "DNIT_005_2003_TER.pdf": f"{BASE}/terminologia-ter/dnit_005_2003_ter-1.pdf",
    "DNIT_006_2003_PRO.pdf": f"{BASE}/procedimento-pro/dnit006_2003_pro.pdf",
    "DNIT_007_2003_PRO.pdf": f"{BASE}/procedimento-pro/DNIT_007_2003_PRO",
    "DNIT_008_2003_PRO.pdf": f"{BASE}/procedimento-pro/DNIT_008_2003_PRO",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; coleta-normas-rag/1.0)"}
DESTINO = Path("docs")
ZIP_SAIDA = Path("normas_dnit_001_008.zip")


def baixar() -> list[Path]:
    DESTINO.mkdir(exist_ok=True)
    arquivos = []
    for nome, url in NORMAS.items():
        caminho = DESTINO / nome
        print(f"Baixando {nome} ...")
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            print(f"  AVISO: {nome} nao parece ser um PDF valido — verifique o link.")
        caminho.write_bytes(resp.content)
        arquivos.append(caminho)
    return arquivos


def zipar(arquivos: list[Path]) -> None:
    with zipfile.ZipFile(ZIP_SAIDA, "w", zipfile.ZIP_DEFLATED) as zf:
        for arq in arquivos:
            zf.write(arq, arcname=arq.name)
    print(f"\nOK: {ZIP_SAIDA} gerado com {len(arquivos)} normas.")


if __name__ == "__main__":
    zipar(baixar())
