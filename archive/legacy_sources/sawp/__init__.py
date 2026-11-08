#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pacote SAWP
-----------

Este pacote reúne várias funções utilitárias relacionadas ao seu projeto.
Todos os módulos são carregados automaticamente para simplificar o uso.

Exemplo:
    import sawp
    sawp.minha_funcao()
"""

from importlib import import_module
import pkgutil

__all__ = []

# Carrega automaticamente todos os módulos dentro de sawp/
for module in pkgutil.iter_modules(__path__):
    nome_modulo = module.name
    mod = import_module(f"{__name__}.{nome_modulo}")

    # Se o módulo definir __all__, importa apenas o que está lá
    if hasattr(mod, "__all__"):
        for item in mod.__all__:
            globals()[item] = getattr(mod, item)
        __all__.extend(mod.__all__)
    else:
        # Caso contrário, importa todas as funções/variáveis públicas
        for item in dir(mod):
            if not item.startswith("_"):  # evita itens privados
                globals()[item] = getattr(mod, item)
                __all__.append(item)

# Versão do pacote
__version__ = "0.1.0"