# Apresentação — 01_completa

Esta versão usa **as imagens que já estão no repositório**. Não possui cópias de PNG.

- Arquivos de apresentação: PDF e PowerPoint (autossuficientes e com imagens incorporadas).
- Fonte editável: arquivo `.tex`, com caminhos relativos às pastas `questao1/` e `questao2/`.
- Roteiro: arquivo `.md`.

## Compilação

Mantenha esta pasta dentro de `entrega/slides/01_completa/` e compile o `.tex` **a partir desta pasta**.
O caminho `../../../` sobe de `entrega/slides/01_completa/` até a raiz do projeto.

Exemplo: `pdflatex -interaction=nonstopmode apresentacao_spambase_completa.tex` (duas execuções para atualizar referências).

As imagens são atualizadas no PDF ao recompilar o LaTeX quando seus arquivos originais mudam. A apresentação já exportada em PDF/PPTX não se atualiza sozinha.
