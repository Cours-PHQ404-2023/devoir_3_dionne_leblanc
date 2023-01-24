""" OHQ_wrapper.py

Implémentation de deux fonctions appliquant la méthode des 
éléments finis et la méthode du tir ainsi que deux fonctions 
permettant de normaliser puis d'afficher les fonctions propres résultantes. 
"""

import numpy as np
import mef
import methode_tir
import scipy.sparse as sp
import matplotlib.pyplot as plt


def normaliseur(valeurs):
    """ Normalise une liste de valeurs au sens de la somme 
    de sa valeur absolue au carré (\sum_i |f_i|^2 = 1)

    Arguments
    ---------
    valeurs : list[float]
        Liste des données à normaliser.

    Retour
    ------
    list[float]
        Les données normalisées. 
    """
    return valeurs/np.sqrt(np.sum(np.square(valeurs)))


def solutions_mef(L, N_points, N_sols):
    """Fonction qui génère les premières solutions à l'équation 
    de Schrodinger pour l'oscillateur harmonique quantique sur 
    une grille unidimensionnelle centrée en 0 grâce à la méthode 
    des éléments finis.

    Arguments
    ---------
    L : int or float
        La demie longueur du système borné en [-L,L].
    N_points : int
        Le nombre de points composant la grille bornée en [-L,L].
    N_sols : int
        Le nombre de premiers états liés à retourner. 

    Retour
    ------
    list(float)
        Les N premières énergies des états liés.
    list(float)
        La grille spatiale utilisée pour la résolution.
    array(float, ndim=2, shape=(N_points, N_sols))
        Les N premiers états propres du système. 
        Un état correspond à une colonne du tableau.
    """
    x_grid = np.linspace(-L, L, N_points) # génère la grille symétrique et uniforme

    elems_finis = mef.Grille(x_grid) # initialisation d'un instance de Grille

    # définition des matrices utiles dans la MEF
    laplacien = elems_finis.matrice_laplacienne_interne() # d_x^2
    potentiel = elems_finis.matrice_potentiel(lambda x: x*x) # x^2
    matrice_masse = elems_finis.matrice_masse_interne()
    matrice_masse_inv = sp.linalg.inv(matrice_masse)

    L_C = (potentiel - laplacien)/2 # Opérateur différentiel (x^2 - d_x^2)/2 
    
    # Calcul des valeurs/vecteur propres
    eigvals, eigvecs = sp.linalg.eigsh(L_C, M=matrice_masse, which="SM", k=N_sols)

    eigvecs_repr_pos = np.empty_like(eigvecs)
    for i in range(len(eigvals)):
        # transformation des vect. propres dans la repr. pos. et normalisation
        eigvecs_repr_pos[:,i] = normaliseur(matrice_masse_inv.dot(eigvecs[:,i]))

    return eigvals, x_grid[1:-1], eigvecs_repr_pos


def solutions_mt(L, N_points, N_sols, vec_0=[0,0.001], val_0=0.2, D=-0.01, rtol_cadrage=0.1, max_iter_cadrage=20, val_max=10):
    """Fonction qui génère les premières solutions à l'équation 
    de Schrodinger pour l'oscillateur harmonique quantique sur 
    une grille unidimensionnelle centrée en 0 grâce à la méthode 
    du tir.

    Arguments
    ---------
    L : int or float
        La demie longueur du système borné en [-L,L].
    N_points : int
        Le nombre de points composant la grille bornée en [-L,L].
    N_sols : int
        Le nombre de premiers états liés à retourner. 
    vec_0 : list(float)
        Les conditions initiales de la fonction d'onde à x=-L 
        sous la forme [psi, psi'].
    val_0 : float
        Valeur initiale du balayage pour le cadrage de racines
        en fonction de l'énergie.
    D : float
        Le pas de l'algoritme de cadrage de racine.
    rtol_cadrage : float
        La tolérance relative employée dans le processus de 
        cadrage de racine pour l'équation f(x_1) = -f(x_0).
    max_iter : int
        Le nombre maximal de pas dans le processus de cadrage 
        avant d'avancer la borne val_0.
    val_max : float
        L'énergie maximale permise pour les solutions.

    Retour
    ------
    list(float)
        Les N premières énergies des états liés.
    list(float)
        La grille spatiale utilisée pour la résolution.
    array(float, ndim=2, shape=(N_points, N_sols))
        Les N premiers états propres du système. 
        Un état correspond à une colonne du tableau.
    """
    x_grid = np.linspace(-L, L, N_points) # génération de la grille
    
    # définition d'une fonction retournant psi(L) en fonction de l'énergie
    BC_de_E = lambda E : methode_tir.calcul_Schrodinger(vec_0, x_grid, E)[-1,0]
    
    # Trouver les premières racines
    racines = methode_tir.trouver_premieres_racines(
        BC_de_E, 
        val_0, 
        N_sols, 
        D=D, 
        rtol=rtol_cadrage, 
        max_iter=max_iter_cadrage, 
        val_max=val_max
    )
    
    # Calcul fonctions propres
    fonctions_propres = np.empty((N_points, len(racines)))

    for i in range(len(racines)):
        # normalisation des fonctions et mise en forme
        fonctions_propres[:,i] = normaliseur(methode_tir.calcul_Schrodinger(vec_0, x_grid, racines[i])[:,0])

    return racines, x_grid, fonctions_propres
        

def faire_graphique(sol, scale, E_roundoff):
    """ Permet de faire un graphique des états propres 
    à partir du retour des fonctions solutions_mt() et 
    solutions_mef().
    
    Arguments
    ---------
    sol : tuple(list[float], list[float], array(float, ndims=2))
        Les premières solution à l'équation de Schrodinger dans 
        le format de retour des fonctions solutions_mt() et 
        solutions_mef().
    scale : float
        Le multiplicateur appliqué aux fonctions propres pour 
        faciliter l'affichage.
    E_roundoff : int
        Le nombre de décimales à garder dans l'affichage de 
        l'énergie.

    Retour
    ------
    None
        Affiche le graphique matplotlib.
    """
    plt.figure() # initialiser la figure

    for i in range(len(sol[0])):
        # faire un graphique de chaque fonction propre décalée par son énergie
        plt.plot(sol[1], scale*sol[2][:,i] + sol[0][i], label = f"$E_{i}$" + " = " + str(round(sol[0][i], E_roundoff)))

    plt.xlabel("x")
    plt.ylabel("$\psi_i + E_i$")
    plt.legend(loc="lower left")

    plt.show() # montrer la figure