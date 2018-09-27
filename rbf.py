import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

class RBF(object):
    @staticmethod
    def f(net):
        return (1/(1+ np.exp(-net)))

    @staticmethod
    def df_dnet(f_net):
        return (f_net*(1 - f_net))

    def kernel_function(self, center, x, sigma):
        return np.exp(-(np.linalg.norm(center-x)**2)/(2*sigma**2))    

    def __init__(self, input_len, centers, hidden_len, output_len, ids):
        # Sigmas
        self.sigmas = None
        # Centroides
        self.centroids = None
        # classes de cada instância
        self.ids = ids

        # Número de neurônios da camada de entrada
        self.input_len = input_len
        # Número de neurônios da camada de centroides
        self.centers = centers
        # Número de neurônios da camada escondida
        self.hidden_len = hidden_len
        # Número de neurônios da camada de saída
        self.output_len = output_len
        # Representação matricial das camadas escondida e de saída
        self.hidden_layer = np.random.rand(self.hidden_len, self.centers+1)
        self.output_layer = np.random.rand(self.output_len, self.hidden_len+1)

        # Variáveis auxiliares para calculos posteriores:
        self.net_h = None # net da camada escondida
        self.f_net_h = None # f(net) da camada escondida
        self.net_o = None # net da camada de saída
        self.f_net_o = None # f(net) da camada de saída

    def forward(self, x):
        # Concatenando 1 que será multiplicado pelo theta
        X = np.concatenate((x, [1]), axis=0)

        # Calculando o net da camada escondida
        self.net_h = np.sum(np.multiply(self.hidden_layer, X), axis=1)
        # f(net) da camada escondida
        self.f_net_h = self.f(self.net_h)

        # Calculando um vetor auxiliar com as multiplicações dos pesos da camada de saída
        # com os f(net) da camada escondida         
        m = np.multiply(self.output_layer, np.concatenate((self.f_net_h, [1]), axis=0))
        # Somando os valores obtidos para se obter o net da camada de saída
        self.net_o = np.sum(m, axis=1)
        # f(net) da camada de saída
        self.f_net_o = self.f(self.net_o)

        return self.f_net_o

    def backwards(self, X, Y, eta=0.1, threshold=1e-3, alpha=0.5, max_iter=2000):
        # Inicializando o erro quadrático com um valor maior que o threshold
        sqerror = 2*threshold
       
        dE2_dw_o = 0
        dE2_dw_h = 0

        dE2_dw_o_t = 0
        dE2_dw_h_t = 0        

        # Contador de iterações
        counter = 0
        while(sqerror > threshold and counter < max_iter):
            # print(counter)
            sqerror = 0

            # Pra cada linha (instância) do dataset
            for i in range(len(X)):
                # Entrada atual
                x = np.array(X[i].flat)
                # print(x)
                # Saída esperada
                y = np.array(Y[i].flat)
                # print(y)
                
                # Calculando a saída da rede neural
                self.forward(x)

                #  Calculando o erro
                error = y - self.f_net_o
                sqerror += np.sum(error*error)

                # Calculando a derivada do erro para a camada de saída
                del_o = error * self.df_dnet(self.f_net_o)

                # Calculando a derivada do erro para a camada escondida
                output_weights = self.output_layer[:,0:self.hidden_len]
                del_h = np.array([self.df_dnet(self.f_net_h) * np.dot(del_o, output_weights)])

                # Atualizando os pesos da rede
                f_net_h = np.concatenate((self.f_net_h, [1]), axis=0)
                dE2_dw_o = np.multiply(np.array([del_o]).T, np.array([f_net_h]))
                dE2_dw_h = np.multiply(del_h.T, np.concatenate((x, [1]), axis=0))

                self.output_layer = self.output_layer + eta * dE2_dw_o + alpha * dE2_dw_o_t
                self.hidden_layer = self.hidden_layer + eta * dE2_dw_h + alpha * dE2_dw_h_t

                dE2_dw_o_t = dE2_dw_o
                dE2_dw_h_t = dE2_dw_h

            sqerror = sqerror / len(X) 
            if(counter % 100 == 0):
                print("sqerror =", sqerror)
            counter += 1

        print("sqerror final =", sqerror)

    
    def calculate_centroids(self, X, centroids):
        ids = self.ids
        self.centroids = []
        # Calculando os centroides por meio da média entre instâncias da mesma classe
        for i in range(centroids):
            self.centroids.append(np.mean(X[ids == i+1], axis=0))

        self.sigmas = []
        # Calculando os sigmas por meio do desvio padrão das instâncias da mesma classe
        for i in range(centroids):
            self.sigmas.append(np.std(X[ids == i+1]))

    def train(self, X, Y):
        # Calculando os centróides
        self.calculate_centroids(X, self.centers)
        
        # Aplicando a kernel function
        x = []
        for i in range(self.centers):
            x.append(np.apply_along_axis(self.kernel_function, 1, X, self.centroids[i], self.sigmas[i]))

        x = np.array(x).T
        self.backwards(x, Y)

        
    def predict(self, X):
        x = []
        for i in range(self.centers):
            x.append(self.kernel_function(X, self.centroids[i], self.sigmas[i]))
        return self.forward(x)

    def test(self, X, Y):
        total = len(X)
        correct = 0
        for i in range(len(X)):
            x = np.array(X[i].flat)
            y = np.array(Y[i].flat)
            y_p = self.predict(x)

            j = np.argmax(y_p)
            
            for k in range(len(y_p)):
                if(k == j):
                    y_p[k] = 1
                else:
                    y_p[k] = 0

            if(np.array_equal(y, y_p) == 1):
                correct += 1

        print("Accuracy:", correct/total)

seeds = pd.read_csv('datasets/seeds_dataset.txt')

columns = seeds.columns
x = seeds[columns[0:columns.size-1]]
y = seeds[columns[columns.size-1]]

X = scale(x)

Y = []
for i in range(y.shape[0]):
    if(y[i] == 1):
        Y.append([1,0,0])
    if(y[i] == 2):
        Y.append([0,1,0])
    if(y[i] == 3):
        Y.append([0,0,1])

Y = np.matrix(Y)

ids = np.random.permutation(len(X))
n = int(0.8 * len(X))
X_train = X[ids[:n]]
Y_train = Y[ids[:n]]
X_test = X[ids[n:]]
Y_test = Y[ids[n:]]

y_aux = np.array(y[ids[:n]])

rbf = RBF(7, 3, 8, 3, y_aux)
rbf.train(X_train, Y_train)
rbf.test(X_test, Y_test)