<h1 align="center"> Devoir #3 Schrodinger </h1>
<p align="center">
Théo Dionne et Jérôme Leblanc
</p>

## Structure

```bash
.
├── OHQ_wrapper.py
├── README.md
├── mef.py
├── mt.py
├── pyproject.toml
└── rapport_Schrodinger.ipynb
```

`rapport_Schrodinger.ipynb` agit comme notre rapport. À l'intérieur s'y trouve le code pertinent et les figures. Le rapport utilise le code définit dans `OHQ_wrapper.py` qui contient les fonctions implémentant la méthode du tir et des éléments finis à un haut niveau ainsi que des fonctions pour la représentation graphique des solutions. Les fonctions utilisées pour implémenter la méthode du tir étape-par-étape se retrouvent dans `mt.py` alors que celles qui sont pertinentes pour la méthode des éléments finis se trouve dans `mef.py`. Cette dernière est issue du dépot contenant l'énoncé du devoir.
