#-------------------------------------------------------------------------------
# Importation des modules

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st
import include as inc

#-------------------------------------------------------------------------------
# Application principale

st.title("Régression linéaire")

st.write("""
    ### Un exemple de régression linéaire (univariée)

""")

# Les données : comme il y en a pas beaucoup nous n'utilisons pas de fichiers
# externes (csv ou autre) pour les stocker.
X = np.array([1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101])
Y = np.array([3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870])

# Affichage des données brutes (sans graphiques) :
msg = "Les données d'apprentissage sont les deux séries de nombres $X$ et $Y$ suivants :"
st.write(msg)
st.latex('''X = (1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101)''')
st.latex('''Y = (3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870)''')

# La régression avec Scikit-Learn :
model, w, b = inc.regression_lineaire(X, Y)
msg = "Cette régression a été faite avec **Scikit-Learn**."
msg = msg + " Le modèle renvoyé par **Scikit-Learn** est le suivant :"
st.write(msg)
st.latex("\\widehat{y} = f(x) = wx + b")
st.latex(f"w={w}\quad b={b} ")

# Représentation graphique :
fig, ax = inc.affichage_regression(X, Y, w, b)
st.pyplot(fig)

# Prédiction pour une nouvelle valeur de x
msg = "Pour faire une prédiction avec une autre valeur de x, saisir un nombre et valider :" 
x = float(st.text_input(msg, 0))

# Prédiction calculée directement à l'aide des paramètres w et b :
y = w*x + b
st.write("La prédiction peut être calculée directement à partir des paramètres de $w$ et $b$ :")
s = f"x={x:.3f} \\quad\\quad \\widehat y ={y:.3f}"
st.latex(s)

# Prédiction calculée avec la méthode 'predict' de Scikit-Learn :
x_vect = np.array([[x]])
ybis = model.predict(x_vect)
# Attention aux types de données Numpy :
ybis = ybis[0][0]

msg = "Évidemment, la prédiction peut aussi être calculée à l'aide de la "
msg = msg + "méthode `predict()` de **Scikit-Learn** : "
st.write(msg)
s = f"x={x:.3f} \\quad\\quad \\widehat y ={ybis:.3f}"
st.latex(s)

# Fin de l'application
st.write("Les résultats sont identiques, ouf ! 😎 ")

#-------------------------------------------------------------------------------
# Conclusion avec le lien vers les sources sur GitHub

st.markdown("""
    <hr>
""", unsafe_allow_html=True)

inc.espace(2)

st.write("""
    📝 Sources de l'application :
    [https://github.com/pbejian/regression_simple](https://github.com/pbejian/regression_simple)

""")
#-------------------------------------------------------------------------------