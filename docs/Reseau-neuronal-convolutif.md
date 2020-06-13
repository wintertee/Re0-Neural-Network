# Réseau neuronal convolutif

## Neurone

Pour un neurone simple, on note:

|  $x$   |                 l'entrée                 |
| :----: | :--------------------------------------: |
|  $w$   |                 le poids                 |
|  $b$   |                 le biais                 |
|  $z$   | la sortie avant la fonction d'activation |
| $\phi$ |         la fonction d'activation         |
|  $a$   | la sortie après la fonction d'activation |

alors on a:
$$
z = w \times x + b \tag{1.1}
$$

$$
a = \phi(z) \tag{1.2}
$$

D'après le théorème de dérivation des fonctions composées:
$$
\frac{\partial a}{\partial x}=\frac{\partial a}{\partial z}\times\frac{\partial z}{\partial x}=\phi'\times w \tag{1.3}
$$

$$
\frac{\partial a}{\partial w}=\frac{\partial a}{\partial z}\times\frac{\partial z}{\partial w}=\phi'\times x \tag{1.4}
$$

$$
\frac{\partial a}{\partial b}=\frac{\partial a}{\partial z}\times\frac{\partial z}{\partial b}=\phi'\times 1 \tag{1.5}
$$

Ce sont les formules fondamentales pour la rétropropagation du gradient des réseaux neuronal.

## Fonction d'activation

Dans la suite, on note $\phi$ une fonction d'activation.

Pour la simplicité, on utilise seulement *unité de rectification linéaire (ReLU)* et *softmax*.

### ReLU

*ReLU* est définie par:
$$
{\displaystyle \phi(a)=\left\{{\begin{array}{rcl}0&{\mbox{si}}&a<0\\a&{\mbox{si}}&a\geq 0\end{array}}\right.}
$$

$$
{\displaystyle \phi'(a)=\left\{{\begin{array}{rcl}0&{\mbox{si}}&a<0\\1&{\mbox{si}}&a\geq 0\end{array}}\right.}
$$

On utilize le sous-gradient pour $a=0$

### Softmax

*Softmax* est définie par:
$$
{\displaystyle a_j = \phi (\mathbf {z} )_{j}={\frac {\mathrm {e} ^{z_{j}}}{\sum _{k=1}^{K}\mathrm {e} ^{z_{k}}}}}\quad\forall j \in \left[ 1,q \right]
$$
Pour sa dérivée partielle, si $j=i$ :
$$
\begin{align*}
\frac{\partial \phi_i}{\partial z_j}&=
\frac{\partial \frac{e^{z_i}}{\sum_{k=1}^{N}e^{z_k}}}{\partial z_j}\\
&=\frac{e^{z_i}(\sum_{k=1}^{N}{e^{z_k}})-e^{z_j}e^{z_i}}{(\sum_{k=1}^{N}{e^{z_k}})^2}\\
&=\frac{e^{z_i}}{\sum_{k=1}^{N}{e^{z_k}}}\frac{(\sum_{k=1}^{N}{e^{z_k}}) - e^{z_j}}{\sum_{k=1}^{N}{e^{z_k}}}\\
&=\phi_i(1-\phi_j)
\end{align*}
$$
si $j\neq i$ :
$$
\begin{align*}
\frac{\partial \phi_i}{\partial z_j}&=
\frac{\partial \frac{e^{z_i}}{\sum_{k=1}^{N}e^{z_k}}}{\partial z_j}\\
&=\frac{0-e^{z_j}e^{z_i}}{(\sum_{k=1}^{N}{e^{z_k}})^2}\\
&=-\frac{e^{z_j}}{\sum_{k=1}^{N}{e^{z_k}}}\frac{e^{z_i}}{\sum_{k=1}^{N}{e^{z_k}}}\\
&=-\phi_j \phi_i
\end{align*}
$$
finalement:
$$
{\displaystyle \frac{\partial\phi_i}{\partial z_j}(\mathbf z)=\left\{{\begin{array}{rcl}\phi_i(1-\phi_j)&{\mbox{si}}&i=j\\-\phi_j\phi_i&{\mbox{si}}&i\neq j\end{array}}\right.}
$$

## Fonction de perte

Dans ce projet, on utilise *entropie croisée* $H$. Soit $\mathbf y \in M_{q,1}$ la vérité-terrain encodée one-hot,  $\mathbf  a$ la sortie de la derrière couche, alors :
$$
{\mathrm  {H}}(\mathbf y,\mathbf a)=-\sum _{j=1}^q y_j\log a_j.\!
$$
Sa dérivée, pour la simplicité, on calcule $\frac{\partial H(\mathbf y , \phi ( \mathbf z))}{\partial z_i}=\frac{\partial H(\mathbf y , \mathbf a)}{\partial z_i}$
$$
\begin{align*}
\frac{\partial H(\mathbf y , \mathbf a)}{\partial z_i}&=
-\sum _{j=1}^q y_j \frac{\partial \log a_j}{\partial z_i}\\
&=-\sum _{j=1}^q y_j \frac{\partial a_j}{a_j \partial z_i}\\
&=-y_i\frac{a_i(1-a_i)}{a_i}-\sum _{j=1,j\neq i}^q y_j \frac{- a_i a_j}{a_j}\\
&=-y_i(1-a_i)+\sum _{j=1,j\neq i}^q y_j a_i\\
&=-y_i +a_i \sum _{j=1}^q y_j\\
&=a_i-y_i
\end{align*}
$$


## Couche linéaire

Pour une couche linéaire, les méthodes sont les même que celles d'un neurone, sauf que les variables sont représentées en forme matricielle.

Pour la  $L$-ième couche dans un réseaux séquentiel dont l'entrée est de dimension $p$, la sortie est de dimension $q$ , les paramètres se représentent en forme matricielle.
$$
\mathbf x^{(L)} \in  M_{p,1}\\
\mathbf W^{(L)} \in  M_{q,p}\\
\mathbf z^{(L)} \in  M_{q,1}\\
\mathbf a^{(L)} \in  M_{q,1}\\
\phi(\mathbf z)=\phi(z_1,z_2,...,z_q)=[\phi(z_1)\quad\phi(z_2)\quad...\quad\phi(z_q)]^T
$$
