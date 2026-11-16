# Apresentação — 04_decisoes_preprocessamento

Esta versão expande e reorganiza a apresentação `03_decisoes`, adicionando:
- **Agenda da apresentação** (Slide 2);
- **Problemas do Double K-means encontrados em relação à base Spambase** (Slide 4);
- **Necessidade de realizar pré-processamento nessa base para superar as limitações do Double K-means** (Slide 6);
- **Reorganização e atualização dos demais slides** para conectar o diagnóstico da base à escolha de `log1p + MinMax`, à estrutura dos co-clusters e à avaliação supervisionada na Questão 2.

Assim como nas demais pastas, esta versão usa **as imagens que já estão no repositório** (`questao1/` e `questao2/`), sem duplicar arquivos PNG.

- Arquivos de apresentação: PDF e PowerPoint (autossuficientes e com imagens incorporadas).
- Fonte editável: `apresentacao_spambase_decisoes_preprocessamento.tex`, com caminhos relativos `../../../questao1/` e `../../../questao2/`.
- Roteiro de apresentação: `roteiro_spambase_decisoes_preprocessamento.md`.

## Compilação

Mantenha esta pasta dentro de `entrega/slides/04_decisoes_preprocessamento/` e compile o `.tex` **a partir desta pasta**:

```bash
pdflatex -interaction=nonstopmode apresentacao_spambase_decisoes_preprocessamento.tex
pdflatex -interaction=nonstopmode apresentacao_spambase_decisoes_preprocessamento.tex
```

O caminho `../../../` sobe de `entrega/slides/04_decisoes_preprocessamento/` até a raiz do projeto.
