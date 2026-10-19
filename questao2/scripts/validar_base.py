"""Execute na raiz do repositório: python -m questao2.scripts.validar_base"""
import json
from questao2.src.dados import validar_base

def main():
    print(json.dumps(validar_base(), ensure_ascii=False, indent=2))
    print('OK: base oficial da Questão 2 validada.')

if __name__ == '__main__':
    main()
