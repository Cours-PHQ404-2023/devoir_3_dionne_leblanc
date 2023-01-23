"""
Implémentation de la méthode du tir.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import odeint


def Schrodinger_RHS(vec, x, E):
    """ Calcule le membre droit de l'équation de Schrodinger en fonction
    du vecteur d'état et de la position.

    Arguments
    ---------
    vec : list[float]
        Le vecteur contenant la fonction d'état et sa dérivée spatiale.
    x : float
        La position spatiale.
    E : float
        L'énergie de la particule.

    Retour
    ------
    Le membre droit de l'équation de Schrödinger
    """

    return [vec[1], (x*x - 2*E)*vec[0]]


def calcul_Schrodinger(vec_0, x_range, E):
    """
    
    """

    return odeint(Schrodinger_RHS, vec_0, x_range, args=(E,))


def generateur_cadre(func, val_0, D=-0.1, rtol=0.1, max_iter=20):
    """
    """
    f_0 = func(val_0) # calcul de la valeur initiale de la fonction
    i = 0

    while i < max_iter:
        i += 1
        val_1 = val_0 + i*D
        f_1 = func(val_1)

        if np.isclose(-f_0, f_1, rtol=rtol):
            return (val_0, val_1)

    return None


def trouver_premiers_cadres(func, val_0,  N, D=-0.01, rtol=0.1, max_iter=20, val_max=10):
    """PUT IN FAILSAFES"""
    liste_cadres = []

    while val_0 < val_max and len(liste_cadres) < 6 :
        cadre = generateur_cadre(func, val_0, D, rtol, max_iter)

        if cadre is not None:
            liste_cadres.append(cadre)

        val_0 += np.abs(D)*max_iter

    return liste_cadres


def trouver_premieres_racines(func, val_0,  N, D=-0.01, rtol=0.1, max_iter=20, val_max=10):
    racines = []
    cadres = trouver_premiers_cadres(func, val_0,  N, D=-0.01, rtol=0.1, max_iter=20, val_max=10)

    for i in range(len(cadres)):
        racines.append(brentq(func, *cadres[i]))

    return racines
    

if __name__ == "__main__":
    print("Running tests, I guess.")