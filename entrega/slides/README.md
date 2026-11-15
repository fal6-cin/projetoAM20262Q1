# Slides — Projeto AM 2026.2 (Spambase)

Três versões da apresentação, com figuras **referenciadas diretamente** dos resultados existentes no projeto:

- `01_completa/`: 37 slides, incluindo apoio técnico.
- `02_principal/`: 23 slides, sem apoio técnico.
- `03_decisoes/`: 11 slides, centrados em decisões e evidências.

## Posicionamento obrigatório

Copie **o conteúdo desta pasta** para `entrega/slides/` na raiz do projeto:

```text
entrega/slides/
  01_completa/
  02_principal/
  03_decisoes/
  ORIGEM_DAS_FIGURAS.csv
  README.md
```

Os arquivos `.tex` usam caminhos relativos como
`../../../questao1/resultados/log_minmax/silhouette_KH.png`.
Eles precisam permanecer nessa estrutura para compilar.

**Não há cópias dos gráficos PNG nesta pasta.** As referências de cada versão constam em `ORIGEM_DAS_FIGURAS.csv`.

Os PDFs e PowerPoints continuam contendo imagens incorporadas: são formatos exportados, independentes do repositório. O PowerPoint utiliza slides renderizados como imagens; edite textos e fórmulas no `.tex`.

Ao mudar um gráfico original, recompile o `.tex` para gerar um novo PDF e reexporte o PowerPoint se necessário.
