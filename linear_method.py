import numpy as np 


"""
Input : Une matrice adjacence 1 A un graphe dirigé, pondéré et régulier G, un vecteur de personnalisation v, ainsi qu un paramètre de téléportation alpha compris entre
0 et 1 (0.9 par défaut et pour les résultats à présenter). Toutes ces valeurs sont nonnégatives.
Output : Un vecteur x contenant les scores dimportance des noeuds ordonnés dans
le même ordre que les lignes de la matrice adjacence (représentant les noeuds).
"""

def pageRankLinear (A : np.matrix , alpha : float, v : np.array) : 
    # Création de la matrice diagonale
    D = np.diag(np.sum(A, axis=0))
    return D

