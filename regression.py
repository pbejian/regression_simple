#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st

#-------------------------------------------------------------------------------

def espace(n):
    """
    Permet de sauter n lignes dans le rendu de streamlit.
    """
    for _ in range(n):
        st.write("")

#-------------------------------------------------------------------------------

st.title("Régression linéaire")

st.write("""
    ### Un exemple de régression linéaire (univariée)

""")
#st.write(" ")

#-------------------------------------------------------------------------------

data = {"X" : [1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101],
        "Y" : [3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870]}

X = np.array([1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101])
Y = np.array([3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870])

#-------------------------------------------------------------------------------

msg = "Cette régression a été faite avec Scikit-Learn."
msg = msg + " Les données d'apprentissage sont les deux séries de nombres X et Y suivant :"
st.write(msg)

st.latex('''X = (1.081, 1.854, 2.674, 3.753, 4.693, 5.498, 6.470, 7.386, 7.981, 9.101)''')
st.latex('''Y = (3.165, 6.047, 4.831, 8.790, 9.266, 14.059, 17.403, 21.370, 21.400, 27.870)''')

#-------------------------------------------------------------------------------

X_train = X.reshape(-1, 1)
Y_train = Y.reshape(-1, 1)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
w = lin_reg.coef_
b = lin_reg.intercept_
w = w[0][0]
b = b[0]
print(f"w={w} b={b}")

#-------------------------------------------------------------------------------
st.write("""
    Le modèle retourné par Scikit-Learn est le suivant : 
""")

st.latex("f(x) = wx + b")
st.latex(f"w={w}\quad b={b} ")

#-------------------------------------------------------------------------------

Z = np.array([0, 10])
T = w*Z + b

#width = st.sidebar.slider("plot width", 1, 25, 3)
#height = st.sidebar.slider("plot height", 1, 25, 1)

width = 2
height = 2

fig, ax = plt.subplots(figsize=(width, height))
plt.tick_params(axis = 'both', labelsize = 3)
#sns.regplot(X, Y, color="r")
#plt.scatter(X, Y, marker="o", color = "b", s=100)
#plt.plot(Z, T, c="r")
ax.scatter(X, Y, marker="o", color = "b", s=3)
ax.plot(Z, T, c="r", linewidth=0.5)
#ax.legend()


#sns.scatterplot(data=df, x="X", y="Y", marker='o', s=500)
#sns.lmplot(data=df, x="X", y="Y", height=10)
st.pyplot(fig)

#-------------------------------------------------------------------------------

msg = "Pour faire une prédiction avec une autre valeur de x, saisir un nombre et valider :" 

x = float(st.text_input(msg, 0))

# Prédiction calculée directement à l'aide des paramètres w et b
y = w*x + b

# Prédiction calculée avec Scikit-Learn (qui demande de vectoriser).
x_vect = np.array([[x]])

z = lin_reg.predict(x_vect)
z = z[0][0]

#-------------------------------------------------------------------------------

st.write("La prédiction peut être calculée directement à partir des paramètres de w et b :")

s = f"x={x:.3f} \\quad\\quad \\widehat y ={y:.3f}"

st.latex(s)

msg = "Évidemment, la prédiction peut aussi être calculée par Scikit-Learn :"

st.write(msg)

s = f"x={x:.3f} \\quad\\quad \\widehat y ={z:.3f}"

st.latex(s)

st.write("Les résultats sont identiques, ouf ! 😎 ")

#-------------------------------------------------------------------------------
st.markdown("""
    <hr>
""", unsafe_allow_html=True)

#-------------------------------------------------------------------------------

espace(2)

st.write("""
📝 Sources de l'application :
[https://github.com/pbejian/hello_tensorflow](https://github.com/pbejian/hello_tensorflow)

""")

#-------------------------------------------------------------------------------