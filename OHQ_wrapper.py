import numpy as np
import mef
import methode_tir
import scipy.sparse as sp
import matplotlib.pyplot as plt


def normaliseur(valeurs):
    return valeurs/np.sqrt(np.sum(np.square(valeurs)))


def solutions_mef(L, N_points, N_sols):
    """
    FAIRE
    """
    x_grid = np.linspace(-L, L, N_points)

    elems_finis = mef.Grille(x_grid)

    laplacien = elems_finis.matrice_laplacienne_interne()
    potentiel = elems_finis.matrice_potentiel(lambda x: x*x)
    matrice_masse = elems_finis.matrice_masse_interne()
    matrice_masse_inv = sp.linalg.inv(matrice_masse)

    L_C = (potentiel - laplacien)/2
    
    eigvals, eigvecs = sp.linalg.eigsh(L_C, M=matrice_masse, which="SM", k=N_sols)

    eigvecs_repr_pos = np.empty_like(eigvecs)
    for i in range(len(eigvals)):
        eigvecs_repr_pos[:,i] = normaliseur(matrice_masse_inv.dot(eigvecs[:,i]))

    return eigvals, x_grid[1:-1], eigvecs_repr_pos


def solutions_mt(L, N_points, N_sols, vec_0, val_0=0.2, D=-0.01, rtol=0.1, max_iter=20, val_max=10):

    x_grid = np.linspace(-L, L, N_points)
    BC_de_E = lambda E : methode_tir.calcul_Schrodinger(vec_0, x_grid, E)[-1,0]

    racines = methode_tir.trouver_premieres_racines(BC_de_E, val_0,  N_sols, D=D, rtol=rtol, max_iter=max_iter, val_max=val_max)

    fonctions_propres = np.empty((N_points, len(racines)))

    for i in range(len(racines)):
        fonctions_propres[:,i] = normaliseur(methode_tir.calcul_Schrodinger(vec_0, x_grid, racines[i])[:,0])

    return racines, x_grid, fonctions_propres
        

def faire_graphique(sol, scale, E_roundoff):
    plt.figure()

    for i in range(len(sol[0])):
        plt.plot(sol[1], scale*sol[2][:,i] + sol[0][i], label = f"$E_{i}$" + " = " + str(round(sol[0][i], E_roundoff)))

    plt.xlabel("x")
    plt.ylabel("$\psi_i + E_i$")
    plt.legend()

    plt.show()