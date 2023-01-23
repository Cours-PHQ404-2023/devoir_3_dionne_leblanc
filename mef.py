"""
Implémentation des principales functions pour la méthode des éléments finis
en une dimension.

Cette implémentation se base sur le chapitre 12 des notes de cours
de David Sénéchal.
"""
import numpy as np
import scipy.sparse as sp
import scipy.integrate as integrate


class Grille:
    """Grille de points pour la méthode des élements finis.

    Arguments
    ---------

    points: Liste ordonnées des points définissant la grille.
    """

    def __init__(self, points):
        # On s'assure que les points sont ordonnées.
        if np.any((points[1:] - points[:-1]) < 0.0):
            raise ValueError("les points ne sont pas ordonnés")
        self.points = points

    def __len__(self):
        """Nombre de points sur la grille."""
        return len(self.points)

    def matrice_masse_interne(self):
        """Retourne la matrice de masse de la grille
        sous forme de scipy.sparse.csc_matrix.

        Les éléments aux frontières ne sont pas calculés.
        Ainsi, l'élément (0, 0) de la matrice retournée
        représente la valeur (1, 1) de la matrice complète.
        """
        n = len(self)
        matrice = sp.dok_matrix((n - 2, n - 2))
        # On calcule les valeurs sur la diagonale centrale.
        for site in range(1, n - 1):
            matrice[site - 1, site - 1] = (
                self.points[site + 1] - self.points[site - 1]
            ) / 3.0
        # On calcule les valeurs sur les autres diagonales.
        for site in range(1, n - 2):
            valeur = (self.points[site + 1] - self.points[site]) / 6.0
            matrice[site - 1, site] = valeur
            matrice[site, site - 1] = valeur
        return matrice.tocsc()

    def matrice_laplacienne_interne(self):
        """Retourne la matrice de l'opérateur différentiel (D^2) de la grille
        sous forme de scipy.sparse.csc_matrix.

        Les éléments aux frontières ne sont pas calculés.
        Ainsi, l'élément (0, 0) de la matrice retournée
        représente la valeur (1, 1) de la matrice complète.
        """
        n = len(self)
        matrice = sp.dok_matrix((n - 2, n - 2))
        # On calcule les valeurs sur la diagonale centrale.
        for site in range(1, n - 1):
            matrice[site - 1, site - 1] = sum(
                1.0 / (self.points[i] - self.points[i + 1])
                for i in [site - 1, site]
            )
        # On calcule les valeurs sur les autres diagonales.
        for site in range(1, n - 2):
            valeur = 1.0 / (self.points[site + 1] - self.points[site])
            matrice[site - 1, site] = valeur
            matrice[site, site - 1] = valeur
        return matrice.tocsc()

    def matrice_potentiel(self, potentiel):
        """Retourne la matrice de potentiel de la grille
        sous forme de scipy.sparse.csc_matrix.

        Les éléments aux frontières ne sont pas calculés.
        Ainsi, l'élément (0, 0) de la matrice retournée
        représente la valeur (1, 1) de la matrice complète.

        Arguments
        ---------

        potentiel: une fonction de float vers float représentant
                   le potentiel à calculer.
        """
        # Calcule les ratios pour les intégrales.
        def pente_pos(x, site):
            num = x - self.points[site - 1]
            denom = self.points[site] - self.points[site - 1]
            return num / denom

        def pente_neg(x, site):
            num = self.points[site + 1] - x
            denom = self.points[site + 1] - self.points[site]
            return num / denom

        n = len(self)
        matrice = sp.dok_matrix((n - 2, n - 2))

        # On calcule les valeurs sur la diagonale centrale.
        for site in range(1, n - 1):
            valeur_pos = integrate.quad(
                lambda x: potentiel(x) * pente_pos(x, site)**2,
                self.points[site - 1],
                self.points[site]
            )[0]
            valeur_neg = integrate.quad(
                lambda x: potentiel(x) * pente_neg(x, site)**2,
                self.points[site],
                self.points[site + 1]
            )[0]
            matrice[site - 1, site - 1] = valeur_pos + valeur_neg

        # On calcule les valeurs sur les autres diagonales.
        for site in range(1, n - 2):
            valeur = integrate.quad(
                lambda x: (
                    potentiel(x)
                    * pente_pos(x, site + 1)
                    * pente_neg(x, site)
                ),
                self.points[site],
                self.points[site + 1]
            )[0]
            matrice[site - 1, site] = valeur
            matrice[site, site - 1] = valeur
        return matrice.tocsr()


if __name__ == "__main__":
    grille = Grille(np.arange(5))

    # Ceci retourne une erreur si les matrices sont différentes.
    np.testing.assert_allclose(
        grille.matrice_masse_interne().toarray(),
        np.array([
            [2/3, 1/6,   0],
            [1/6, 2/3, 1/6],
            [  0, 1/6, 2/3],
        ])
    )

    # Ceci retourne une erreur si les matrices sont différentes.
    np.testing.assert_allclose(
        grille.matrice_laplacienne_interne().toarray(),
        np.array([
            [-2,  1,  0],
            [ 1,  -2,  1],
            [ 0,   1, -2],
        ])
    )

    # Ceci retourne une erreur si les matrices sont différentes.
    np.testing.assert_allclose(
        grille.matrice_potentiel(lambda x: x**2).toarray(),
        np.array([
            [0.733, 0.383,     0],
            [0.383, 2.733,  1.05],
            [    0,  1.05, 6.067],
        ]),
        atol=1e-3,
    )
