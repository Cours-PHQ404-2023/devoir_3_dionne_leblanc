""" methode_tir.py

Fonctions permettant l'implémentation de la méthode du tir
pour trouver les états propres d'un oscillateur harmonique 
quantique (OHQ).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import odeint


def Schrodinger_RHS(vec, x, E):
    """ Calcule le membre droit (au sens du format requis par les résolveurs 
    d'équations différentielles) de l'équation de Schrodinger en fonction
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
    array(float, ndims=1, shape=(2))
        Le membre droit de l'équation de Schrödinger
    """

    return [vec[1], (x*x - 2*E)*vec[0]]


def calcul_Schrodinger(vec_0, x_range, E):
    """ Calcule la solution à l'équation de Schrodinger 
    étant donné les conditions initiales pour une énergie 
    spécifiée sur une grille finie.
    
    Arguments
    ---------
    vec_0 : list(float)
        Les conditions initiales de la fonction d'onde à x=-L 
        sous la forme [psi, psi'].
    x_range : list(float)
        La grille sur laquelle solutionner l'équation de Schrodinger.
    E : float
        Énergie pour laquelle résoudre l'équation de Schrodinger.

    Retour
    ------
    array(float, ndims=2, shape=(len(x_range), 2))
        La fonction d'onde et sa dérivée spatiale en tout point de la grille.
    """

    return odeint(Schrodinger_RHS, vec_0, x_range, args=(E,))


def generateur_cadre(func, val_0, D=-0.1, rtol=0.1, max_iter=20):
    """ Fonction qui retourne un cadre pour une racine de 
    f(x) si la condition f(x_1) = -f(x_0) est respectée à 
    l'intérieur d'une certaine tolérance.

    Arguments
    ---------
    func : function --> float
        Une fonction scalaire pour laquelle on cherche à 
        cadrer une racine.
    val_0 : float
        Valeur initiale pour le cadre.
    D : float
        Incrément de progression de la deuxième borne.
        (NOTE: Le signe affecte le déplacement de la borne.)
    rtol : float
        Tolérance relative dans la condition f(x_1) = -f(x_0)
    max_iter : int
        Le nombre maximal de pas dans le processus de cadrage 
        avant d'avancer la borne val_0.

    Retour
    ------
    tuple(size=2) or None
        Retourne un cadre sous forme de tuple si détecté,
        sinon, retourne None.
    """
    f_0 = func(val_0) # calcul de la valeur initiale de la fonction
    i = 0 # initialise le compteur

    while i < max_iter:
        i += 1 # incrémente
        val_1 = val_0 + i*D # met à jour la valeur de val_1
        f_1 = func(val_1) # fonction évaluée à x_1

        # retourne un cadre si la condition f(x_1) = -f(x_0) est respectée (dans l'intervalle de tolérance relative rtol)
        if np.isclose(-f_0, f_1, rtol=rtol): 
            return (val_0, val_1) # retourne le cadre

    return None


def trouver_premiers_cadres(func, val_0,  N, D=-0.01, rtol=0.1, max_iter=20, val_max=10):
    """ Fonction qui cadre les N premières racines.
    
    Arguments
    ---------
    func : function --> float
        Une fonction scalaire pour laquelle on cherche à 
        cadrer une racine.
    val_0 : float
        Valeur initiale du balayage pour le cadrage de racines
        en fonction de l'énergie.
    N : int
        Nombre de solutions à trouver.
    D : float
        Le pas de l'algoritme de cadrage de racine.
    rtol : float
        La tolérance relative employée dans le processus de 
        cadrage de racine pour l'équation f(x_1) = -f(x_0).
    max_iter : int
        Le nombre maximal de pas dans le processus de cadrage 
        avant d'avancer la borne val_0.
    val_max : float
        L'énergie maximale permise pour les solutions.

    Retour
    ------
    list[tuple(size=2)]
        Retourne les cadres des racines.
    """
    liste_cadres = [] # initialise liste de cadres

    # Boucle autant que le nombre de cadres n'est pas trouvé ou que la valeur maximale n'est pas atteinte
    while val_0 < val_max and len(liste_cadres) < N :
        # Essaie de trouver un cadre dans l'intervalle |D|*max_iter
        cadre = generateur_cadre(func, val_0, D, rtol, max_iter)

        if cadre is not None:
            # Si le cadre existe, il est rajouté à la liste
            liste_cadres.append(cadre)

        val_0 += np.abs(D)*max_iter # incrémente la région de recherche

    if val_max == val_0:
        # Soulève une erreur si la valeur maximale est atteinte
        raise RuntimeError(f"Valeur maximale de E={val_max}")

    return liste_cadres


def trouver_premieres_racines(func, val_0,  N, D=-0.01, rtol=0.1, max_iter=20, val_max=10):
    """ Permet de trouver les racines d'une fonction en 
    cadrant d'abord ses solutions avec trouver_premiers_cadres.
    
    Arguments
    ---------
    func : function --> float
        Une fonction scalaire pour laquelle on cherche à 
        cadrer une racine.
    val_0 : float
        Valeur initiale du balayage pour le cadrage de racines
        en fonction de l'énergie.
    N : int
        Nombre de solutions à trouver.
    D : float
        Le pas de l'algoritme de cadrage de racine.
    rtol : float
        La tolérance relative employée dans le processus de 
        cadrage de racine pour l'équation f(x_1) = -f(x_0).
    max_iter : int
        Le nombre maximal de pas dans le processus de cadrage 
        avant d'avancer la borne val_0.
    val_max : float
        L'énergie maximale permise pour les solutions.

    Retour
    ------
    list[float]
        Les racines de la fonction `func`.
    """
    racines = [] # initialize la liste de racines
    
    # Essaie de trouver les N premiers cadres
    cadres = trouver_premiers_cadres(func, val_0,  N, D=D, rtol=rtol, max_iter=max_iter, val_max=val_max)

    for i in range(len(cadres)):
        # Trouve la racine dans chaque cadre
        racines.append(brentq(func, *cadres[i]))

    return racines