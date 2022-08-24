import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

def espace(n):
    """
    Cette fonction ne renvoie rien mais affiche n lignes vides
    dans une application streamlit.
    """
    for _ in range(n):
        st.write("")
    return None

def regression_lineaire(X, Y):
    """
    In  : Deux tableaux Numpy composé de flottants et de taille shape=(n,) 
    Out : Une liste [model, w , b] de telles sorte que 'model' soit une
          instance de LinearRegression() entraîné avec les données X et Y.
          Les coefficients w et b sont les nombres tels que le modèle 
          corresponde à la fonction f(x) = wx + b
    """
    # On commence par redimensionner les données :
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    # On crée un modèle de régression linéaire et on l'entraîne avec X et Y
    model = LinearRegression()
    model.fit(X, Y)
    # On récupère les paramètres du modèle (attention ce sont des tableaux)
    w = model.coef_
    b = model.intercept_
    w = w[0][0]
    b = b[0]
    # On renvoie le modèle et ses coefficients (flottants) sous forme de tuple.
    return (model, w, b)

def affichage_regression(X, Y, w, b):
    """
    In  : Deux tableaux Numpy composés de flottants et de taille shape=(n,),
          ainsi que les deux paramètres w et b pour lesquel la droite
          de régression a pour équation y = wx + b. 
    Out : Renvoie un couple (fig, ax) qui est une représentation graphique
          de la régression linéaire basée sur les données de X et Y (affichage
          des points et de la droite de régression).
          Remarque - Cela devra ensuite être passé à st.pyplot()    
    """
    # Configuration de l'affichage
    fig, ax = plt.subplots(figsize=(2, 2))
    plt.tick_params(axis = 'both', labelsize = 3)
    # Les points
    ax.scatter(X, Y, marker="o", color = "b", s=3)
    # La droite de régression
    Z = np.array([0, 10])
    T = w*Z + b
    ax.plot(Z, T, c="r", linewidth=0.5)
    return (fig, ax)
