"""Ablacao EXPLORATORIA de pre-processamento da Questao 2.

Nao modifica os quatro avaliadores originais nem seus CSVs 30x10.
Mantem as mesmas divisoes externas, as mesmas sementes internas e a mesma
selecao por F1 macro. Scalers sempre ajustados somente no treino interno.
"""

from __future__ import annotations

import json
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .avaliacao_gaussiano import REGULARIZACOES
from .avaliacao_knn import VALORES_K, DISTANCIAS
from .avaliacao_logistica import VALORES_C, PESOS_CLASSES
from .avaliacao_parzen import VALORES_H, _predicoes_para_grade
from .dados import ALVOS, carregar_dados
from .metricas import calcular_metricas
from .modelos.bayes_gaussiano import BayesGaussianoMultivariado
from .modelos.bayes_knn import BayesKVizinhos
from .modelos.bayes_parzen import BayesParzen
from .modelos.regressao_logistica import RegressaoLogistica
from .modelos.voto_majoritario import VotoMajoritario
from .validacao import gerar_folds_externos, gerar_folds_internos

VARIANTES = ("sem", "minmax", "log1p_minmax")
MODELOS_BASE = ("gaussiano", "knn", "parzen", "logistica")
# Log-espaçada para dados NA ESCALA ORIGINAL (não usar grade calibrada em [0,1]).
H_BRUTO = (0.05, 0.2, 1.0, 5.0, 20.0, 100.0, 500.0)

# Grade ampliada exclusivamente para Parzen com MinMaxScaler.
H_MINMAX_REVISADO = (0.005, 0.01, 0.02, 0.03, 0.04, 0.08, 0.12, 0.20, 0.35, 0.55)

# Cenários reavaliados no estudo complementar.
CENARIOS_REVISADOS = {
    ("sem", "logistica"),
    ("minmax", "parzen"),
}


def transformar(treino: np.ndarray, validacao: np.ndarray, variante: str):
    """Transforma usando parametros fitados exclusivamente em `treino`."""
    if variante not in VARIANTES:
        raise ValueError(f"Variante invalida: {variante}")
    treino = np.asarray(treino, dtype=float)
    validacao = np.asarray(validacao, dtype=float)
    if variante == "sem":
        return treino, validacao
    if variante == "log1p_minmax":
        if (treino < 0).any() or (validacao < 0).any():
            raise ValueError("log1p exige dados nao negativos")
        treino, validacao = np.log1p(treino), np.log1p(validacao)
    escalador = MinMaxScaler().fit(treino)
    return escalador.transform(treino), escalador.transform(validacao)


def pipeline(modelo, variante):
    if variante not in VARIANTES:
        raise ValueError("Variante invalida")
    etapas = []
    if variante == "log1p_minmax":
        etapas.append(("log1p", FunctionTransformer(np.log1p, validate=True)))
    if variante != "sem":
        etapas.append(("minmax", MinMaxScaler()))
    etapas.append(("modelo", modelo))
    return Pipeline(etapas)


def ajustar_gaussiano(X, y, semente, variante):
    modelo = pipeline(BayesGaussianoMultivariado(), variante)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    busca = GridSearchCV(
        modelo,
        {"modelo__regularizacao_relativa": REGULARIZACOES},
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    busca.fit(X, y)
    return busca.best_estimator_, busca.best_params_, float(busca.best_score_)


def ajustar_logistica(X, y, semente, variante):
    if variante == "sem":
        classificador = RegressaoLogistica(
            max_iter=20000,
            tol=1e-6,
        )
    else:
        classificador = RegressaoLogistica()

    modelo = pipeline(classificador, variante)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=semente,
    )

    busca = GridSearchCV(
        modelo,
        {
            "modelo__C": VALORES_C,
            "modelo__class_weight": PESOS_CLASSES,
        },
        scoring="f1_macro",
        cv=cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )

    busca.fit(X, y)

    return (
        busca.best_estimator_,
        busca.best_params_,
        float(busca.best_score_),
    )


def ajustar_knn(X, y, semente, variante):
    classes = np.unique(y)
    pontos = {(d, k): [] for d in DISTANCIAS for k in VALORES_K}
    for tr, va in gerar_folds_internos(y, semente=semente):
        Xt, Xv = transformar(X[tr], X[va], variante)
        for distancia in DISTANCIAS:
            modelo = BayesKVizinhos(n_vizinhos=max(VALORES_K), distancia=distancia).fit(
                Xt, y[tr]
            )
            indices = modelo.indices_vizinhos(Xv, k=max(VALORES_K))
            cod = modelo.y_codificado_[indices]
            for k in VALORES_K:
                cod_prev = BayesKVizinhos._vencedores_com_desempate(
                    cod[:, :k], len(classes)
                )
                prev = modelo.classes_[cod_prev]
                pontos[(distancia, k)].append(
                    float(
                        f1_score(
                            y[va],
                            prev,
                            labels=classes,
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
    escolhas = [
        (float(np.mean(pontos[(d, k)])), d, k)
        for d in DISTANCIAS
        for k in sorted(VALORES_K)
    ]
    esc = max(escolhas, key=lambda x: x[0])  # desempate por ordem conhecida
    estimador = pipeline(BayesKVizinhos(n_vizinhos=esc[2], distancia=esc[1]), variante)
    estimador.fit(X, y)
    return estimador, {"distancia": esc[1], "k": int(esc[2])}, esc[0]


def ajustar_parzen(X, y, semente, variante):
    classes = np.unique(y)

    if variante == "sem":
        larguras = H_BRUTO
    elif variante == "minmax":
        larguras = H_MINMAX_REVISADO
    else:
        larguras = VALORES_H

    pontos = {float(h): [] for h in larguras}

    for tr, va in gerar_folds_internos(y, semente=semente):
        Xt, Xv = transformar(X[tr], X[va], variante)
        d2 = cdist(Xv, Xt, metric="sqeuclidean")
        codigos = np.searchsorted(classes, y[tr])

        predicoes = _predicoes_para_grade(d2, codigos, classes, larguras)

        for h, pred in predicoes.items():
            pontuacao = f1_score(
                y[va],
                pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
            pontos[float(h)].append(float(pontuacao))

    escolhas = sorted((float(np.mean(valores)), h) for h, valores in pontos.items())

    maior_f1 = max(media for media, _ in escolhas)

    # Mantém o desempate do experimento original.
    melhor_h = min(
        h for media, h in escolhas if np.isclose(media, maior_f1, atol=1e-14, rtol=0)
    )

    estimador = pipeline(
        BayesParzen(largura_janela=melhor_h),
        variante,
    )

    estimador.fit(X, y)

    return (
        estimador,
        {
            "h": melhor_h,
            "grade_h": list(larguras),
        },
        maior_f1,
    )


AJUSTADORES = {
    "gaussiano": ajustar_gaussiano,
    "knn": ajustar_knn,
    "parzen": ajustar_parzen,
    "logistica": ajustar_logistica,
}


def _salvar_json_atomico(caminho: Path, dados: dict):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=caminho.parent, delete=False
    ) as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2, allow_nan=False)
        temporario = arquivo.name
    os.replace(temporario, caminho)


def _arquivo(raiz, alvo, variante, rep, fold, modelo):
    revisado = (variante, modelo) in CENARIOS_REVISADOS or (
        modelo == "voto" and variante in ("sem", "minmax")
    )

    pasta_execucao = "execucoes_ajustes_v2" if revisado else "execucoes"

    return (
        raiz
        / pasta_execucao
        / alvo
        / f"rep_{rep:02d}_fold_{fold:02d}"
        / f"{variante}_{modelo}.json"
    )


def executar(
    raiz: Path,
    *,
    alvo: str = "todos",
    variante: str = "todas",
    modelos: tuple = MODELOS_BASE,
    repeticoes: int = 1,
    max_folds: int | None = None,
    semente: int = 2026,
    verbose=True,
):
    """Executa comparacao pareada e salva resultados POR MODELO para retomada.

    `max_folds` e destinado apenas a verificacao; jamais confundir com estudo completo.
    Variantes compartilham fold exterior e semente da validacao interna.
    """
    if repeticoes < 1 or (max_folds is not None and max_folds < 1):
        raise ValueError("Repeticoes e max_folds devem ser positivos")
    alvos = ALVOS if alvo == "todos" else (alvo,)
    variantes = VARIANTES if variante == "todas" else (variante,)
    if not set(alvos).issubset(ALVOS) or not set(variantes).issubset(VARIANTES):
        raise ValueError("Alvo ou variante desconhecidos")
    if not set(modelos).issubset(MODELOS_BASE):
        raise ValueError("Modelo desconhecido")
    X, ys, _ = carregar_dados()
    raiz = Path(raiz)
    for a in alvos:
        y = ys[a]
        folds = gerar_folds_externos(
            y, repeticoes=repeticoes, folds=10, semente=semente
        )
        for idx, (rep, fold, tr, te) in enumerate(folds, 1):
            if max_folds is not None and idx > max_folds:
                break
            for var in variantes:
                for nome in modelos:
                    arquivo = _arquivo(raiz, a, var, rep, fold, nome)
                    if arquivo.exists():
                        anterior = json.loads(arquivo.read_text(encoding="utf-8"))
                        if (
                            len(anterior.get("predicoes", [])) == len(te)
                            and anterior.get("indices_teste") == te.tolist()
                            and anterior.get("sementes")
                            == {
                                "cv_externa": semente,
                                "cv_interna": semente + 1000 * rep + fold,
                            }
                        ):
                            if verbose:
                                print(
                                    f"[salvo] {a} {var} r{rep} f{fold} {nome}",
                                    flush=True,
                                )
                            continue
                        raise RuntimeError(
                            f"Arquivo existente inconsistente: {arquivo}"
                        )
                    if verbose:
                        print(f"[rodar] {a} {var} r{rep} f{fold} {nome}", flush=True)
                    with warnings.catch_warnings(record=True) as alertas:
                        warnings.simplefilter("always", ConvergenceWarning)
                        estimador, parametros, cv = AJUSTADORES[nome](
                            X[tr], y[tr], semente + 1000 * rep + fold, var
                        )
                    numero_alertas_convergencia = sum(
                        issubclass(w.category, ConvergenceWarning) for w in alertas
                    )
                    if numero_alertas_convergencia and verbose:
                        print(
                            f"[ATENCAO] {numero_alertas_convergencia} alertas de nao convergencia em {nome}/{var}; interpretar com cautela",
                            flush=True,
                        )
                    previsoes = estimador.predict(X[te])
                    m = calcular_metricas(y[te], previsoes)
                    _salvar_json_atomico(
                        arquivo,
                        {
                            "alvo": a,
                            "variante": var,
                            "repeticao": rep,
                            "fold": fold,
                            "modelo": nome,
                            "indices_teste": te.tolist(),
                            "predicoes": np.asarray(previsoes, dtype=int).tolist(),
                            "n_treino": len(tr),
                            "n_teste": len(te),
                            "parametros": parametros,
                            "avisos_convergencia": int(numero_alertas_convergencia),
                            "f1_macro_cv_interna": cv,
                            "sementes": {
                                "cv_externa": semente,
                                "cv_interna": semente + 1000 * rep + fold,
                            },
                            "metricas": {
                                k: v for k, v in m.items() if k != "por_classe"
                            },
                        },
                    )
                # voto é reconstituído estritamente dos mesmos exemplos externos
                voto_arq = _arquivo(raiz, a, var, rep, fold, "voto")
                arqs = [_arquivo(raiz, a, var, rep, fold, k) for k in MODELOS_BASE]
                if all(p.exists() for p in arqs) and not voto_arq.exists():
                    objs = [json.loads(p.read_text(encoding="utf-8")) for p in arqs]
                    if any(obj["indices_teste"] != te.tolist() for obj in objs):
                        raise RuntimeError(
                            "Predicoes dos classificadores nao alinhadas"
                        )
                    matriz = np.array([obj["predicoes"] for obj in objs])
                    votacao = VotoMajoritario().fit(y[tr])
                    pred = votacao.predict(matriz)
                    m = calcular_metricas(y[te], pred)
                    _salvar_json_atomico(
                        voto_arq,
                        {
                            "alvo": a,
                            "variante": var,
                            "repeticao": rep,
                            "fold": fold,
                            "modelo": "voto",
                            "indices_teste": te.tolist(),
                            "predicoes": pred.astype(int).tolist(),
                            "n_treino": len(tr),
                            "n_teste": len(te),
                            "metricas": {
                                k: v for k, v in m.items() if k != "por_classe"
                            },
                        },
                    )
    return consolidar(
        raiz, alvos=alvos, variantes=variantes, repeticoes=repeticoes, semente=semente
    )


def consolidar(
    raiz: Path, *, alvos=ALVOS, variantes=VARIANTES, repeticoes=1, semente=2026
):
    """Publica tabelas SOMENTE quando 10 folds x repeticoes x 5 modelos completos."""
    raiz = Path(raiz)
    X, ys, _ = carregar_dados()
    linhas = []
    for alvo in alvos:
        y = ys[alvo]
        for rep, fold, tr, te in gerar_folds_externos(
            y, repeticoes=repeticoes, folds=10, semente=semente
        ):
            for variante in variantes:
                for modelo in (*MODELOS_BASE, "voto"):
                    arquivo = _arquivo(raiz, alvo, variante, rep, fold, modelo)
                    if not arquivo.exists():
                        return {
                            "completo": False,
                            "faltando": str(arquivo),
                            "total_parcial": len(linhas),
                        }
                    dado = json.loads(arquivo.read_text(encoding="utf-8"))
                    if dado["indices_teste"] != te.tolist() or len(
                        dado["predicoes"]
                    ) != len(te):
                        raise ValueError(f"Indices invalidos em {arquivo}")
                    metr = calcular_metricas(y[te], np.asarray(dado["predicoes"]))
                    for k in (
                        "f_measure_macro",
                        "taxa_erro",
                        "precisao_macro",
                        "cobertura_macro",
                    ):
                        if not np.isclose(
                            metr[k], dado["metricas"][k], rtol=0, atol=1e-12
                        ):
                            raise ValueError(f"Metrica inconsistente {arquivo} {k}")
                    linhas.append(
                        {
                            "alvo": alvo,
                            "variante": variante,
                            "modelo": modelo,
                            "repeticao": rep,
                            "fold": fold,
                            "n_teste": len(te),
                            "avisos_convergencia": int(
                                dado.get("avisos_convergencia", 0)
                            ),
                            **{
                                k: metr[k]
                                for k in (
                                    "f_measure_macro",
                                    "taxa_erro",
                                    "precisao_macro",
                                    "cobertura_macro",
                                )
                            },
                        }
                    )
    df = pd.DataFrame(linhas)
    resumo = (
        df.groupby(["alvo", "modelo", "variante"], sort=True)[
            [
                "f_measure_macro",
                "taxa_erro",
                "precisao_macro",
                "cobertura_macro",
                "avisos_convergencia",
            ]
        ]
        .agg(
            {
                "f_measure_macro": ["mean", "std"],
                "taxa_erro": ["mean", "std"],
                "precisao_macro": ["mean", "std"],
                "cobertura_macro": ["mean", "std"],
                "avisos_convergencia": ["sum"],
            }
        )
        .reset_index()
    )
    resumo.columns = [
        "_".join(filter(None, col)) if isinstance(col, tuple) else col
        for col in resumo.columns
    ]
    raiz.mkdir(parents=True, exist_ok=True)
    df.to_csv(raiz / "comparacao_folds.csv", index=False)
    resumo.to_csv(raiz / "comparacao_resumo.csv", index=False)
    return {
        "completo": True,
        "linhas": len(df),
        "resumo": str(raiz / "comparacao_resumo.csv"),
    }
