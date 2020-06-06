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
{\displaystyle \Phi (\mathbf {a} )_{j}={\frac {\mathrm {e} ^{a_{j}}}{\sum _{k=1}^{K}\mathrm {e} ^{a_{k}}}}}\quad\forall j \in \left[ 1,q \right]
$$
si $j=i$ :
$$
\begin{align*}
\frac{\partial \Phi_i}{\partial a_j}&=
\frac{\partial \frac{e^{a_i}}{\sum_{k=1}^{N}e^{a_k}}}{\partial a_j}\\
&=\frac{e^{a_i}(\sum_{k=1}^{N}{e^{a_k}})-e^{a_j}e^{a_i}}{(\sum_{k=1}^{N}{e^{a_k}})^2}\\
&=\frac{e^{a_i}}{\sum_{k=1}^{N}{e^{a_k}}}\frac{(\sum_{k=1}^{N}{e^{a_k}}) - e^{a_j}}{\sum_{k=1}^{N}{e^{a_k}}}\\
&=\Phi_i(1-\Phi_j)
\end{align*}
$$
si $j\neq i$ :
$$
\begin{align*}
\frac{\partial \Phi_i}{\partial a_j}&=
\frac{\partial \frac{e^{a_i}}{\sum_{k=1}^{N}e^{a_k}}}{\partial a_j}\\
&=\frac{0-e^{a_j}e^{a_i}}{(\sum_{k=1}^{N}{e^{a_k}})^2}\\
&=-\frac{e^{a_j}}{\sum_{k=1}^{N}{e^{a_k}}}\frac{e^{a_i}}{\sum_{k=1}^{N}{e^{a_k}}}\\
&=-\Phi_j \Phi_i
\end{align*}
$$
finalement:
$$
{\displaystyle \frac{\partial\Phi_i}{\partial a_j}(\mathbf a)=\left\{{\begin{array}{rcl}\Phi_i(1-\Phi_j)&{\mbox{si}}&i=j\\-\Phi_j\Phi_i&{\mbox{si}}&i\neq j\end{array}}\right.}
$$

## Fonction de perte

Dans ce projet, on utilise *entropie croisée* , définie par :
$$
{\mathrm  {H}}(p,q)=-\sum _{x}p(x)\,\log q(x).\!
$$
未完成

## Couche linéaire

Pour la  $L$-ième couche dans un réseaux séquentiel dont l'entrée est de dimension $p$, la sortie est de dimension $q$ , les paramètres se représentent en forme matricielle.
$$
\mathbf x^{(L)} \in  M_{p,1}\\
\mathbf W^{(L)} \in  M_{q,p}\\
\mathbf z^{(L)} \in  M_{q,1}\\
\mathbf a^{(L)} \in  M_{q,1}\\
\Phi(\mathbf z)=\phi(z_1,z_2,...,z_q)=[\phi(z_1)\quad\phi(z_2)\quad...\quad\phi(z_q)]^T
$$
Les dérivations s'écritent sous forme matricielle:
$$
\frac{\partial \Phi}{\partial \mathbf z}= \Phi'\cdot
$$
未完成
