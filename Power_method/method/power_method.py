import numpy as np

"""
Input : Une matrice adjacence A un graphe dirigé, pondéré et régulier G, un vecteur de personnalisation v, ainsi qu un paramètre de téléportation α compris entre 0
et 1 (0.9 par défaut et pour les résultats à présenter).
Output : Un vecteur x contenant les scores d importance des noeuds ordonnés dans
le même ordre que les lignes de la matrice d adjacence (représentant les noeuds).
"""


def diagonal(A : np.matrix): 
    """Cette fonction prend une matrice d'adjacence A en entrée et retourne une matrice diagonale D.
    La matrice diagonale D a pour éléments les degrés entrants de chaque nœud.
    Elle est utilisée dans le calcul de la matrice de transition."""
    row, column = np.shape(A)
    D = np.zeros((row, column))
    for i in range(len(A)):
        sum_row = np.sum(A[i,:])  # la somme de chaque ligne respective représente une valeur propre dans la matrice diagonale
        D[i, i] = sum_row
    D = np.matrix(D)
    return D

def inverseD(D : np.matrix) : 
    # Cette fonction prend une matrice diagonale D en entrée et retourne son inverse.
    return np.linalg.inv(D)

def transpose(P: np.matrix) : 
    # Cette fonction prend une matrice P en entrée et retourne sa transposée.
    return np.transpose(P)

def transitionmatrix(A : np.matrix, D: np.matrix) :
    # Cette fonction prend une matrice d'adjacence A et une matrice diagonale D en entrée
    # et retourne la matrice de transition P.
    inv_D = inverseD(D)
    
    P = transpose(A) * inv_D

    


def count_node(A : np.matrix) :
    # Cette fonction compte le nombre de nœuds dans le graphe représenté par la matrice d'adjacence A    
    return len(A)

def ones_matrix(A: np.matrix):
    """Cette fonction prend une matrice d'adjacence A en entrée et retourne une matrice remplie de 1/n,
    où n est le nombre de nœuds du graphe."""
    n = count_node(A)
    rows, cols = np.shape(A)
    one = np.ones((rows, cols))
    return 1/n * one

    

def googlematrix(A : np.matrix , P : np.matrix, alpha : float) : 
    """Cette fonction prend une matrice d'adjacence A, une matrice de transition P et un paramètre alpha en entrée
    et retourne la matrice de Google G."""

    G = (alpha * P) + ((1 - alpha) * ones_matrix(A))
    return G

def first_four_personalized_vectors(G : np.matrix, v :np.array): 
    """Cette fonction prend une matrice de Google G et un vecteur de personnalisation v en entrée
     et affiche les quatre premiers vecteurs de personnalisation obtenus par la méthode de puissance."""
    for i in range(1,5) : 
        u = G * v
        v = u
        print("Le vecteur de personnalisation " + f"{i}" + " est égal : " + "\n" + f"{v}" )

def score(G : np.matrix, v :np.array): 
    """Cette fonction prend une matrice de Google G et un vecteur de personnalisation v en entrée
    et retourne le vecteur final des scores d'importance après convergence."""
    max_iteration = 0

    row, column = np.shape(v) 

    u = np.ones((row,column))

    ε =  10**-8 # marge d'erreur toleree 

    while np.linalg.norm(u-v) >= ε : # verifie si la distance euclidienne est strictement superieur a la marge d erreur (= epsilon) 
        if(max_iteration < 100 ) :
            if (max_iteration == 0) : 
                u = G * v  
                max_iteration +=1 
            else : 
                v = u 
                u = G * v 
                max_iteration +=1 
        else : 
            raise RuntimeError("Maximum Number of itération exceeded")
        
    print("Le score final a convergé à la " + f"{max_iteration - 1}"+ "-ème itération valant : " "\n" + f"{v}")

    return v

def pageRankPower(A : np.matrix , alpha : float , v : np.array ): 
    """
    Cette fonction prend une matrice d'adjacence A, un paramètre alpha et un vecteur de personnalisation v en entrée
    et effectue le calcul du PageRank en utilisant la méthode de puissance.
    """

    D = diagonal(A)

    P = transitionmatrix(A, D)

    print("Probability matrix : " +"\n"+  f"{P}")

    G = googlematrix(A, P, alpha)

    print("Google matrix : " +"\n"+  f"{G}")

    f = first_four_personalized_vectors(G , v)

    s = score(G, v)

    return s
    


