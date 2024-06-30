import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

seed = 777
rng = np.random.default_rng(seed)

def rng_int():
    return rng.integers(1, 10000)


def plot_learning_curve(estimator, X, y, cv, scoring='accuracy'):
    """
    
    """
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title("Curva de Aprendizado")
    plt.xlabel("Tamanho do Conjunto de Treinamento")
    plt.ylabel("Acurácia")
    plt.ylim(0.0, 1.1)
    plt.grid()
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='c', label='Pontuação de Treinamento')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='m', label='Pontuação de Validação') 
    
    plt.legend(loc="best")
    plt.show()