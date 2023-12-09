import unittest
from Power_method.method.power_method import count_node,transpose, googlematrix,first_four_personalized_vectors,score
import numpy as np

class TestPM(unittest.TestCase) : 

    def test_countnode(self) :

        # premier test tiré du TP3-exercice supplémentaire 1    
        A = np.matrix([[0,2,0],[0,0,2],[1,1,0]])
        self.assertEqual(count_node(A),3)

        # deuxieme test tiré du lien https://cs.uwaterloo.ca/~kogeddes/cs370/materials/PageRank/PageRank.html
        A = np.matrix([[0,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,1,1,1],[1,0,0,0,0,0],[0,0,0,0,0,1],[1,0,0,0,0,0]])
        self.assertEqual(count_node(A),6)


    def test_googlematrix(self) : 

        # premier test tiré du TP3-exercice supplémentaire
        A = np.matrix([[0,2,0],[0,0,2],[1,1,0]])
        G =np.array([[0,0,0.5],[1,0,0.5],[0,1,0]])
        P = np.matrix([[0,0,0.5],[1,0,0.5],[0,1,0]])
        alpha = 1
        self.assertEqual(np.round(googlematrix(A,P,alpha),2).tolist(), np.round(G,2).tolist())

        # deuxieme test tiré du lien https://cs.uwaterloo.ca/~kogeddes/cs370/materials/PageRank/PageRank.html
        P = np.matrix([[0, 0, 0, 1, 0, 1],[1, 0, 0, 0, 0, 0],[0, 0.5, 0, 0, 0, 0],[0, 0.5, 0.33333333, 0, 0, 0],[0, 0, 0.33333333, 0, 0, 0],[0, 0, 0.33333333, 0, 1, 0]])
        alpha = 0.85
        A = np.matrix([[0,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,1,1,1],[1,0,0,0,0,0],[0,0,0,0,0,1],[1,0,0,0,0,0]])
        G = np.matrix([[0.025, 0.025, 0.025, 0.875, 0.025, 0.875],[0.875, 0.025, 0.025, 0.025, 0.025, 0.025],[0.025, 0.45, 0.025, 0.025, 0.025, 0.025],[0.025, 0.45, 0.30833333, 0.025, 0.025, 0.025],[0.025, 0.025, 0.30833333, 0.025, 0.025, 0.025],[0.025, 0.025, 0.30833333, 0.025, 0.875, 0.025]])
        
        # Arrondir les valeurs à la troisième décimale
        G = np.round(G, 3)
        res = np.round(googlematrix(A,P,alpha),3)
        self.assertEqual(res.tolist(), G.tolist())


    def test_first_four_personalized_vectors(self): 
        
        # premier test tiré du TP3-exercice supplémentaire pour alpha 1 
        G =np.matrix([[0,0,0.5],[1,0,0.5],[0,1,0]])
        v = np.array([[1/6],[3/6],[2/6]])

        expected_output = np.array([[[0.16666667],[0.33333333],[0.5]] ,[[0.25],[0.41666667],[0.33333333]], [[0.16666667],[0.41666667],[0.41666667]], [[0.20833333],[0.375],[0.41666667]] ])  # Replace with your expected output
        for i in range(1, 5):
            u = G * v
            v = u 
            self.assertEqual(np.round(v,8).tolist(),expected_output[i-1].tolist())

        # deuxieme test tiré du lien https://cs.uwaterloo.ca/~kogeddes/cs370/materials/PageRank/PageRank.html
        G = np.matrix([[0.025, 0.025, 0.025, 0.875, 0.025, 0.875],[0.875, 0.025, 0.025, 0.025, 0.025, 0.025],[0.025, 0.45, 0.025, 0.025, 0.025, 0.025],[0.025, 0.45, 0.30833333, 0.025, 0.025, 0.025],[0.025, 0.025, 0.30833333, 0.025, 0.025, 0.025],[0.025, 0.025, 0.30833333, 0.025, 0.875, 0.025]])
        v = np.array([[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]])

        expected_output2 = np.array([[[0.30833333], [0.16666667], [0.09583333], [0.14305556], [0.07222222], [0.21388889]],
                                    [[0.32840278], [0.28708333], [0.09583333], [0.12298611], [0.05215278], [0.11354167]]
                                    ,[[0.22604861], [0.30414236], [0.14701042], [0.17416319], [0.05215278], [0.09648264]], 
                                    [[0.25504896], [0.21714132], [0.1542605], [0.19591345], [0.06665295], [0.11098281]]])

        for i in range(1, 5):
            u = G * v
            v = u
            self.assertEqual(np.round(v,8).tolist(),np.round(expected_output2[i-1],8).tolist())


        


    def test_score(self): 

        # premier test tiré du TP3-exercice supplémentaire
        s = np.array([[1/5],[2/5],[2/5]])
        G =np.matrix([[0,0,0.5],[1,0,0.5],[0,1,0]])
        v = np.array([[1/6],[3/6],[2/6]])
        self.assertEqual(s.tolist(),np.round(score(G,v),8).tolist())

        # deuxieme test tiré du lien https://cs.uwaterloo.ca/~kogeddes/cs370/materials/PageRank/PageRank.html
        s = np.array([[0.267528086917422026],[0.2523988680],[0.1322695192536],[0.1697458847959],[0.062476365542],[0.11558127545]])
        G = np.matrix([[0.025, 0.025, 0.025, 0.875, 0.025, 0.875],[0.875, 0.025, 0.025, 0.025, 0.025, 0.025],[0.025, 0.45, 0.025, 0.025, 0.025, 0.025],[0.025, 0.45, 0.30833333, 0.025, 0.025, 0.025],[0.025, 0.025, 0.30833333, 0.025, 0.025, 0.025],[0.025, 0.025, 0.30833333, 0.025, 0.875, 0.025]])
        v = np.array([[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]])
        self.assertEqual(np.round(s,7).tolist(),np.round(score(G,v),7).tolist())

        

if __name__ == "__main__" : 
    unittest.main()