"""Execute na raiz: python -m questao2.scripts.verificar_folds"""
import json
from questao2.src.dados import carregar_dados
from questao2.src.validacao import conferir_folds, gerar_folds_externos, gerar_folds_internos

def main():
    _, ys, _ = carregar_dados()
    for alvo, y in ys.items():
        resumo=conferir_folds(y)
        r, f, treino, teste = next(gerar_folds_externos(y))
        internos=list(gerar_folds_internos(y[treino]))
        assert len(internos)==5
        resumo['treino_primeiro_fold']=len(treino)
        resumo['teste_primeiro_fold']=len(teste)
        resumo['folds_internos']=len(internos)
        print(alvo, json.dumps(resumo,ensure_ascii=False))
    print('OK: 30×10-folds e 5-folds internos validados para as duas respostas.')

if __name__ == '__main__':
    main()
