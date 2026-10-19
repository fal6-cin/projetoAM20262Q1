"""Validação/consolidação e gráficos para as curvas F1 macro de aprendizagem."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from .curvas_aprendizagem import PERCENTUAIS, TODOS_MODELOS

ROTULOS = {
    'gaussiano': 'Bayes gaussiano',
    'knn': 'Bayes k-vizinhos',
    'parzen': 'Bayes Parzen',
    'logistica': 'Regressão logística',
    'voto': 'Voto majoritário',
}
COLS_F1 = ('f1_macro_treino', 'f1_macro_teste')


def consolidar_curvas(pasta: str | Path, alvos=('y_original', 'y_double_kmeans'),
                      repeticoes: int = 1, exigir_completo: bool = True):
    """Só gera o resumo completo após verificar todos os pontos e modelos."""
    pasta = Path(pasta)
    if repeticoes < 1:
        raise ValueError('Repetições deve ser >=1')
    partes = []
    faltantes = []
    for alvo in alvos:
        for rep in range(1, repeticoes+1):
            for pct in PERCENTUAIS:
                arquivo = pasta/'execucoes'/alvo/f'rep_{rep:02d}_pct_{pct:02d}.csv'
                if not arquivo.exists():
                    faltantes.append(str(arquivo)); continue
                trecho = pd.read_csv(arquivo)
                esperado = set(TODOS_MODELOS)
                if len(trecho) != 5 or set(trecho.modelo) != esperado:
                    raise ValueError(f'Arquivo com modelos ausentes/repetidos: {arquivo}')
                if not ((trecho.alvo == alvo).all() and
                        (trecho.repeticao == rep).all() and
                        (trecho.percentual_treino == pct).all()):
                    raise ValueError(f'Metadados diferentes do caminho: {arquivo}')
                if (trecho.n_treino + trecho.n_teste != 4601).any():
                    raise ValueError(f'Tamanhos incorretos em {arquivo}')
                for c in COLS_F1:
                    if (not np.isfinite(trecho[c].to_numpy()).all() or
                            (trecho[c].to_numpy() < 0).any() or
                            (trecho[c].to_numpy() > 1).any()):
                        raise ValueError(f'F1 fora de [0,1] em {arquivo}')
                partes.append(trecho)
    if faltantes and exigir_completo:
        raise ValueError(f'Curvas INCOMPLETAS: {len(faltantes)} pontos faltando. '
                         f'Exemplo: {faltantes[0]}')
    if not partes:
        raise ValueError('Nenhum ponto da curva calculado')
    bruto = pd.concat(partes, ignore_index=True)
    sufixo = '_PARCIAL' if faltantes else ''
    arquivo_bruto = pasta/f'curvas_pontos_consolidados{sufixo}.csv'
    bruto.to_csv(arquivo_bruto, index=False)
    resumo = (bruto.groupby(['alvo','modelo','percentual_treino'], as_index=False)
              .agg(n_repeticoes=('repeticao', 'count'),
                   f1_treino_medio=('f1_macro_treino', 'mean'),
                   f1_treino_desvio=('f1_macro_treino', 'std'),
                   f1_teste_medio=('f1_macro_teste', 'mean'),
                   f1_teste_desvio=('f1_macro_teste', 'std'),
                   n_treino_medio=('n_treino', 'mean'), n_teste_medio=('n_teste', 'mean')))
    resumo[['f1_treino_desvio','f1_teste_desvio']] = resumo[['f1_treino_desvio','f1_teste_desvio']].fillna(0)
    resumo.to_csv(pasta/f'resumo_curvas{sufixo}.csv', index=False)
    if not faltantes:
        (pasta/'resumo_curvas_PARCIAL.csv').unlink(missing_ok=True)
        (pasta/'curvas_pontos_consolidados_PARCIAL.csv').unlink(missing_ok=True)
    if not faltantes:
        gerar_graficos(resumo, pasta/'graficos')
    return resumo, faltantes


def gerar_graficos(resumo: pd.DataFrame, destino: str | Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    saidas = []
    for alvo in sorted(resumo.alvo.unique()):
        parte = resumo[resumo.alvo == alvo]
        for conjunto, coluna in (('treinamento','f1_treino_medio'), ('teste','f1_teste_medio')):
            fig, ax = plt.subplots(figsize=(10, 6))
            for nome in TODOS_MODELOS:
                linhas = parte[parte.modelo==nome].sort_values('percentual_treino')
                ax.plot(linhas.percentual_treino, linhas[coluna], marker='o', markersize=3,
                        linewidth=1.8, label=ROTULOS[nome])
            ax.set(title=f'Curva de aprendizagem — {alvo} — {conjunto}',
                   xlabel='Percentual dos dados utilizado no treinamento (%)',
                   ylabel='F1 macro', xlim=(5,95),ylim=(0,1.02))
            ax.set_xticks(PERCENTUAIS)
            ax.grid(alpha=0.25)
            ax.legend(loc='lower right', fontsize=9)
            fig.tight_layout()
            saida=destino/f'{alvo}_f1_{conjunto}.png'
            fig.savefig(saida,dpi=165)
            plt.close(fig)
            saidas.append(saida)
        # Curvas pareadas treino/teste por classificador: mostram sub/sobreajuste.
        for nome in TODOS_MODELOS:
            linhas = parte[parte.modelo==nome].sort_values('percentual_treino')
            fig, ax = plt.subplots(figsize=(9,5.5))
            ax.plot(linhas.percentual_treino, linhas.f1_treino_medio, 'o-', label='Treinamento', linewidth=2)
            ax.plot(linhas.percentual_treino, linhas.f1_teste_medio, 's--', label='Teste', linewidth=2)
            ax.set(title=f'{ROTULOS[nome]} — {alvo}',xlabel='% de treinamento',ylabel='F1 macro',
                   xlim=(5,95),ylim=(0,1.02))
            ax.set_xticks(PERCENTUAIS);ax.grid(alpha=0.25);ax.legend()
            fig.tight_layout()
            saida=destino/f'{alvo}_{nome}_treino_vs_teste.png'
            fig.savefig(saida,dpi=165);plt.close(fig);saidas.append(saida)
    return saidas
