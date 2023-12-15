import numpy as np 



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

    if  np.sum(P[:,0]) != 1 :
        return transpose(P)
    else: 
        return P


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



def unit_vector(A : np.matrix): 

    rown , column = np.shape(A)

    e = np.ones(column)

    return e

def identity_matrix(A : np.matrix): 

    I = np.identity(len(A))
    
    return I


def gaussian_matrix(G : np.matrix, I : np.matrix , e : np.array) :
     
    row, column = np.shape(G) 

    S = np.zeros((row+1,column))

    S[0, :] = e

    C = I - G 

    for i in range(1,len(S)): 
        S[i,:] = C[i-1,:]


    S = np.matrix(S)

    return S



def gaussian_vector(v : np.array,  alpha : float) : 

    row , column = np.shape(v) 

    b = np.zeros((row+1,column))

    b[0,:] = 1 
    
    for element in range(1,len(b)):

        b[element,0] = (1-alpha) * v[element-1,0]

    
    print("gaussian vector equal : " + "\n" +  f"{b}")

    return b


def gaussian_elimination(L : np.matrix , b : np.array): 

    if len(L[0:]) != len(b): 
        raise ValueError("Error according to the size of the dimension")
    
    augmented_matrix = np.concatenate((L, b), axis=1, dtype=float)  # Transpose b to match dimensions

    print("La matrice augmentée est égal : " + "\n" + f"{augmented_matrix}")

    rows, cols = augmented_matrix.shape
    x = np.zeros(rows)

    for i in range(rows):
        # Recherche du pivot non nul dans la colonne courante
        pivot_index = i
        while pivot_index < rows and augmented_matrix[pivot_index, i] == 0:
            pivot_index += 1

        # Échanger les lignes si le pivot est nul
        if pivot_index < rows:
            augmented_matrix[[i, pivot_index]] = augmented_matrix[[pivot_index, i]]
        else:
            break
        # Normalisation du pivot à 1
        augmented_matrix[i] /= augmented_matrix[i, i]

        # Élimination des éléments au-dessus et en dessous du pivot
        for j in range(rows):
            if i != j:
                ratio = augmented_matrix[j, i]
                augmented_matrix[j] -= ratio * augmented_matrix[i]


        # Vérifier si deux lignes sont identiques
        if np.all(augmented_matrix[i+1:] == augmented_matrix[i:]):
            print("Deux lignes identiques après élimination de Gauss. Arrêt.")
            break

    if np.all(augmented_matrix[-1] == 0):
        augmented_matrix = augmented_matrix[:-1]


    print("Matrice après étape de Gauss : \n", augmented_matrix)

    return augmented_matrix



def score(augmented_matrix : np.matrix): 

    row,column = np.shape(v) 

    u = np.zeros((row,column))
    
    for element in range(len(augmented_matrix[0:])): 
        u[element,0] = augmented_matrix[element,-1]
        
    return u

def pageRankLinear (A : np.matrix , alpha : float , v : np.array ):
    """
    :param A : Une matrice d adjacence A d un graphe dirigé, pondéré et régulier G, 
    :param v : un vecteur de personnalisation v
    :param alpha : un paramètre de téléportation α compris entre 0 et 1 (0.9 par défaut et pour les résultats à présenter). 
    Toutes ces valeurs sont nonnégatives.
    :return : Un vecteur x contenant les scores d importance des noeuds ordonnés dans
    le même ordre que les lignes de la matrice d adjacence (représentant les noeuds)
    """
    
    D = diagonal(A)

    P = transitionmatrix(A, D)

    print("Probability matrix : " +"\n"+  f"{P}")

    G = googlematrix(A, P, alpha)

    print("Google matrix : " +"\n"+  f"{G}" + "\n")

    I = identity_matrix(A)

    e = unit_vector(A)

    L = gaussian_matrix(G,I,e)

    print("gaussian matrix : " + "\n" + f"{L}")

    b = gaussian_vector(v, alpha)

    R = gaussian_elimination(L,b)

    S = score(R)

    return S

    



   
A = np.matrix([[0,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,1,1,1],[1,0,0,0,0,0],[0,0,0,0,0,1],[1,0,0,0,0,0]])
v = np.array([[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]])
alpha = 0.85


print(pageRankLinear(A,alpha,v))
