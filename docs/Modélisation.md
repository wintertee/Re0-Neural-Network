# INDEX

[toc]

<div style="page-break-after: always; break-after: page;"></div>

# Abstract

En étudiant un modèle d’optimisation de BCD (Block Coordinate Descent), on va étudier la structuré d’apprentissage profond (CNN) et ses propriétés, puis on va faire l’analyse théorique et algorithmique de ce modèle pour établir mathématiquement le problème dual et connaître sa convergence, ses avantages et ses inconvénients etc.

**Mots clés : Réseau d'Apprentissage Profond(CNN), Block Coordinate Descent(BCD), Optimisation, analyse algorithmique** 

<div style="page-break-after: always; break-after: page;"></div>

# 1 Modélisation

## 1.1 Objectif

Quand on voit un nombre, on peut en général facilement le distinguer même si l’on écrit mal. Donc est-ce que c’est possible de faire connaître les numéro à l’ordinateur? Avec l’aide d’un réseau neuronal de convolution(CNN), la réponse est possible: on peut créer une fonction où l’importation est une image de numéro et l’exportation est le résultat de distinction de 0 à 9, même si peut-être on commence par un taux de précision très bas, on peut améliorer les paramètres de cette fonction pour avoir une fonction qui permet de distinguer les numéro correctement.

## 1.2 Solution : réseau neuronal de convolution (CNN)

### 1.2.1 Structure

Un complet CNN comporte plusieurs couches qui ont de différents fonctionnement : couche de convolution et couche de full-confection. Et couche de convolution comporte de plus une couche de pool.

Pour un CNN, on doit aussi avoir une fonction de perte (loss) pour avoir un retour sur le taux de précision de la fonction. De plus, les fonctions d’activation qui permettent de rendre ce système non linéaire et un optimiseur qui permet de évoluer les paramètres de la fonction sont aussi nécessaires.

### 1.2.2 Notations

Tout d’abord, on va donner les notations pour un neurone de CNN:

Pour un neurone simple, on note:

**Table  1 - Notations des paramètres et des variables**

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

De plus, si on voit $ W ^{(l)}$, c’est-à-dire une matrice de tous les poids dans la $l$-ième couche, $ W ^{(l)}_{(n)}$ est le $n$-ième poids dans la $l$-ième couche.

### 1.2.3 Fonction de perte

Dans ce projet, on utilise *entropie croisée* $H$. Soit $\mathbf y \in M_{q,1}$ la vérité-terrain encodée one-hot,  $\mathbf  a$ la sortie de la derrière couche, alors :
$$
{\mathrm  {H}}(\mathbf y,\mathbf a)=-\sum _{j=1}^q y_j\log a_j.\!
$$
Elle est clairement différentiable et on peut avoir sa dérivée, pour la simplicité, on calcule $\frac{\partial H(\mathbf y , \phi ( \mathbf z))}{\partial z_i}=\frac{\partial H(\mathbf y , \mathbf a)}{\partial z_i}$
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

### 1.2.4 Fonction d'activation

Dans la suite, on note $\phi$ une fonction d'activation.

Pour la simplicité, on utilise seulement *unité de rectification linéaire (ReLU)* et *softmax*.

#### 1.2.4.1 ReLU

*ReLU* est définie par:
$$
{\displaystyle \phi(x)=\left\{{\begin{array}{rcl}0&{\mbox{si}}&x<0\\x&{\mbox{si}}&x\geq 0\end{array}}\right.}
$$

$$
{\displaystyle \phi'(x)=\left\{{\begin{array}{rcl}0&{\mbox{si}}&x<0\\1&{\mbox{si}}&x\geq 0\end{array}}\right.}
$$

On utilize le sous-gradient pour $x=0$

Dans les contraintes, on a pour tout $x$ dans tous les couches $x>0$, donc on peut avoir directement:
$$
\phi(x) = x\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\
\phi'(x) = 1
$$


#### 1.2.4.2 Softmax

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

### 1.2.5 Couches

#### 1.2.5.1 Couche de convolution (Conv2D)

La couche de convolution est à la base du réseau CNN. Grâce à un filtre (noyau de convolution), nous pouvons extraire les caractéristiques de l'image. Afin de conserver autant que possible toutes les caractéristiques de l'image, nous envisageons généralement d'utiliser deux couches de convolution.

Une couche de convolution contient deux opérations: propagation vers l'avant (forward) et propagation vers l'arrière (backward)：

**Propagation forward**

Soit un filtre de taille $n\times n$, l’image d’entrée de taille $N\times N$, on a le résultat de cet image après ce filtre:

$Soit \, (i,j)\in \mathbb{N}^2)\,\,\,\,\,\forall (i,j)\in \left [ 1,N-n+1 \right ]\\$
$$
z_{i,j}=\sum _{a\in[1,n],b\in[1,n]}x_{i+a-1,j+b-1}\times w_{a,b}+b_{a,b}
$$
où $w\,\, et\,\, b$ construisent un filtre.

Et après une fonction d’activation:
$$
a_{i,j} = \phi(z_{i,j})
$$
Pour une couche de convolution, elle peut avoir plusieurs filtres.

**Propagation backward**

Dans cette partie, on veut calculer le gradient de chaque poids comportant dans le filtre par l’incertitude définie par:
$$
\delta _{i,j} = \frac{\partial H}{\partial x_{i,j}}
$$
donc on a le gradient $\eta_{i,j}$ correspond à chaque $w_{i,j}$:
$$
\eta _{i,j}=\frac{\partial H}{\partial w_{i,j}}=\sum _m\sum _n\delta _{m,n}\times a_{i+m,j+n}
$$
Aussi pour le biais:
$$
\frac{\partial H}{\partial  b }=\sum _i\sum _j\delta _{i,j}
$$

#### 1.2.5.2 Couche de MaxPooling (MaxPool2D)

On utilise dans ce projet MaxPooling pour faire le “downsampling”: d'une manière générale, en réduisant les caractéristiques pour obtenir l'effet de réduire le nombre de paramètres.

Une couche de MaxPooling comporte aussi “forward” et “backward” mais sans fonction d’activation:

**Propagation forward**

Soit la taille d’un MaxPool est $n\times n$,  l’image d’entrée de taille $N\times N$:
$$
a_{i,j} = max(x_{i,j},...,x_{i,j+n-1},x_{i+1,j},...,x_{i+n-1,j+n-1})
$$
En tout cas, on prend la valeur maximale pour chaque carré de taille $n \times n$ dans l’image d’entrée pour former une nouvelle matrice.

**Propagation backward**

Il est difficile de décrire par les formes mathématiques:

* Premièrement, on prend l’image d’entrée et on garde tous les valeurs qu’on prend dans la propagation forward, et on rend les autres valeurs 0.
* Deuxièmement, on prend les incertitudes données par la propagation backward de la couche prochaine, on remplaces les valeurs qu’on garde dans la première étape une par une par ces incertitudes, on obtient donc le résultat de propagation backward de cette couche de MaxPooling

Un exemple est donné pour illustrer cette propagation:



<img src="https://i.loli.net/2020/06/14/SU5GjHPkCWrOpFn.png" alt="MaxPooling" style="zoom:20%;" />

<center>Figure 1 - Propagation de MaxPooling</center>


#### 1.2.5.3 Exemples des images transformées par les couches de convolution

Pour bien comprendre ce qui s’est passé pour une image transformée par les couches de convolution et de MaxPooling, on va tout d’abord voir directement les résultats:

<img src="https://i.loli.net/2020/06/21/AqZONbiRMU8C3sf.png" alt="compare_conv" style="zoom: 20%;" />

<center>Figure 2 - Visualisation de CNN</center>

Après une série de conversions, nous avons constaté que la taille de l'image est passée de 28$\times$28 à    13$\times$13. Parce que le noyau convolutif est générées de manière aléatoire, de sorte que vous pouvez voir que certains résultats sont bons mais d'autres sont mauvais. Mais dans tous les cas, la plupart des résultats conservent les caractéristiques des numéros originaux.

En général, le rôle principal de la couche de convolution est de minimiser la quantité d'informations dans l'image tout en préservant les caractéristiques des numéro, de sorte que moins de données sont entrées dans la couche linéaire

#### 1.2.5.4 Couche linéaire

Pour une couche linéaire, les méthodes sont les même que celles d'un neurone, sauf que les variables sont représentées en forme matricielle.

Pour la  $L$-ième couche dans un réseaux séquentiel dont l'entrée est de dimension $p$, la sortie est de dimension $q$ , les paramètres se représentent en forme matricielle.
$$
\mathbf x^{(L)} \in  M_{p,1}\\
\mathbf W^{(L)} \in  M_{q,p}\\
\mathbf z^{(L)} \in  M_{q,1}\\
\mathbf a^{(L)} \in  M_{q,1}\\
\phi(\mathbf z)=\phi(z_1,z_2,...,z_q)=[\phi(z_1)\quad\phi(z_2)\quad...\quad\phi(z_q)]^T
$$

Il existe aussi une cohue de Flatten qui permet de transformer une matrice de taille $n\times m $ en une matrice de taille $mn \times 1$ et qui est très facilement à réaliser.

## 1.3 Problème d’optimisation

Pour un réseau neuronal, on a un problème d’optimisation comme:
$$
\begin{aligned}
&\min _{\left(W_{l}, b_{l}\right)_{0}^{L},\left(x_{l}\right)_{1}^{L}} \mathcal L\left(y, X_{L+1}\right)+\sum_{l=0}^{L} \rho_{l} \pi_{l}\left(W_{l}\right)\\
&\text { s.t. } \quad X_{l+1}=\phi_{l}\left(W_{l} X_{l}+b_{l} \mathbf{1}_{m}^{T}\right), \quad l=0, \ldots, L\\
&X_{0}=X
\end{aligned}
$$
avec $\mathcal L$ fonction de perte, $\rho \in \mathbb{R}_{+}^{L+1}$ est un vecteur de hyper-paramètre de la rigularisation, $X$ sont les variables artificiels introduits par BCD, $\phi$ est la fonction d’activation, $L$ le nombre des couches pour ce réseau et $\pi_l$ sont des fonctions qui permettent de représenter les contraintes et la structure du réseau. Ici, $\pi_l$ présente la rigularisation:
$$
\pi_{l}(W)=\|W\|_{F}^{2}, l=0, \ldots, L
$$
et pour les fonctions ReLu, on les représentent par:
$$
\phi(u)=\max (0, u)=\arg \min _{v \geq 0}\|v-u\|_{2}
$$
et pour les conditions $X_{l+1}=\phi_{l}\left( W _{l} X_{l}+b_{l} \mathbf{1}_{m}^{T}\right)$, on a:
$$
X_{l+1} \in \arg \min _{z \geq 0}\left\|z-W_{l} X_{l}-b_{l} \mathbf{1}^{T}\right\|_{}^{2}
$$
donc le problème d’optimisation devient sous la forme pour une certaine couche $l$:
$$
\begin{aligned}\min _{\left(W_{l}, b_{l}\right),\left(X_{l}\right)} & \mathcal L\left(Y,\phi_L( W_{L} X_{L}+b_{L} \mathbf{1}^{T})\right)+\sum_{l=0}^{L} \rho_{l}\left\|W_{l}\right\|_{F}^{2} \\&+\sum_{l=0}^{L-1}\left(c_{l+1}\left\|X_{l+1}-W_{l} X_{l}-b_{l} \mathbf{1}^{T}\right\|_{}^{2}\right) \\\text { s.t. } & X_{l} \geq 0, l=1, \ldots, L-1, X_{0}=X\end{aligned}
$$
où $c_{l+1}>0$ sont les hyper-paramètres.

### 1.3.1 Introduction de BCD

On introduit l’optimiseur BCD: Block Coordinate Descent qui permet de réaliser ce problème d’optimisation.

En générale, on fait renouveler tous les variables de réseau, mais par BCD, pendant chaque propagation il y a seulement un ou certains “blocks” va renouveler.

L’algorithme de BCD qu’on a utilisé dans cet article est un prolongement de RCD (Radom Coordinate Descent), d’après la propriété de propagation de CNN, cette façon de BCD qui permet de renouveler toutes les couches une par une est préférable et plus logique.

On introduit aussi les variables artificiels $X$ pour chaque couche, ce sont les sorties de chaque couche mais qui ne dépendent pas $ W $ ou $b$, donc on rendre un problème de réseau des plusieurs problèmes pour chaque couche, et cela permet de faire devenir un problème non convexe par plusieurs problèmes convexes. On va présenter l’algorithme de BCD un peu plus tard.

### 1.3.2 Propriétés de l’optimisation

Comme on a introduit les variables artificiels, on peut avons les conclusions ci-dessous:

* Convexité: la fonction de perte est une fonction convexe
* Linéarité: avec les fonctions d’activations, ce problème devient un problème non-linéaire.
* Différentiabilité: fonction de perte est différentiable

On va surtout vérifier la convexité de cette optimisation.

#### Convexité

D'après un article 《VISUALIZE THE LOSS LANDSCAPE OF NEURAL NETS》 sur l’étude de la fonction de perte, on voit bien que la convexité de la fonction d’un réseau dépend de batch size, learning rate, optimiseur.

![image-20200705163723490](https://i.loli.net/2020/07/05/9rFstR2SzqTjmd8.png)

<center>Figure 3 - Visualisation de Loss</center>

Si on n’utilise pas le technique skip connections, on voit bien qu’une fonction est clairement non convexe pour un réseau avec plusieures couche.

On va vérifier notre réseau par la méthode Monte Carlo qui va prendre aléatoirement les valeurs de la fonction de perte pour vérifier la convexité.

En fixant tous les poids sauf un noté $ W ^*$, on prend les valeurs de $ W ^*$ aléatoirement pour calculer les valeurs correspondantes  de loss, et on obtient la figure:

<img src="https://i.loli.net/2020/06/25/vJUfawDbFBPjZA6.png" alt="凸性" style="zoom:50%;" />

<center>Figure 4 - Fonction de perte selon un ω fixé</center>

Pour cette figure, l’axe horizontal est les valeurs de $ W ^*$, et l’axe vertical est celles de fonction de perte.

On voit bien que pour la fonction de perte de ce réseau de CNN, la situation est très compliquée et elle est évidemment non convexe.

Donc ici, on introduit les variables artificielles:
$$
\begin{aligned}\min _{\left(W_{l}, b_{l}\right),\left(X_{l}\right)} & \mathcal L\left(Y,\phi_L( W_{L} X_{L}+b_{L} \mathbf{1}^{T})\right)+\sum_{l=0}^{L} \rho_{l}\left\|W_{l}\right\|_{F}^{2} \\&+\sum_{l=0}^{L-1}\left(c_{l+1}\left\|X_{l+1}-W_{l} X_{l}-b_{l} \mathbf{1}^{T}\right\|_{}^{2}\right) \\\text { s.t. } & X_{l} \geq 0, l=1, \ldots, L-1, X_{0}=X\end{aligned}
$$
On considére que les variables $X$ sont artificielles donc ne dépendent pas de $ W $ ni de $b$, dans ce cas, le modèle d’optimisation devient convexe.

# 2 Analyse Dual

## 2.1 Problème duale

### 2.1.1 Notation

$w_i^{(l)} \in \Bbb{R}$, paramètre du CNN

$w^{(l)}$Matrice des paramètres de la couche $l$

$b^{(l)}= [b_1^{(1)},b_2^{(1)},..,b_n^{(l)}]^t $, vecteur des biais de la couche  $l$

$L$, nombre de la couche  $a^{(l)}$, activation de la couche   $l$

$z^{(l+1)} \mathop=\limits^{def} w^{(l)}\cdot a^{(l)} + b^{(l)} $ et $a^{(l)} = \phi(z^{(l)}), l \neq L$, avec $\phi$ la function d’activation”

$l(a^{(L)},w^{L}): \Bbb{R}^n \to \Bbb{R} $  Loss : function de perte

 ### 2.1.2 Position du problème 

En introduisant $X^{(l)}, \forall l \in [0,L]$ les variables artificielles 2 à 2 indépendantes on en déduit que:

$(P)$ 
$$
min \space l(y,\phi^{(L)}(w^{(L)}X^{(L)}+b^{(L)}))+ \sum\limits_{l=0}^{L-1} \lambda_{l+1} || X^{(l+1)} - w^{(l)}\cdot X^{(l)}-b^{(l)}||^2 \\s.c.X^{(l)} \ge 0\space\space \forall l\in [1,L-1], et \space X^{(0)} = X
$$




posons

$g(X)= [X^{(0)},X^{(1)},...,X^{(L)}]^T$ $h(X^{(0)})= X^{(0)}-X$

$(P)$ devient 
$$
f(w,X)=\space l(y,\phi^{(L)}(w^{(L)}X^{(L)}+b^{(L)}))+ \sum\limits_{l=0}^{L-1} c_{l+1} || X^{(l+1)} - w^{(l)}\cdot X^{(l)}-b^{(l)}||^2 + \sum\limits_{l=0}^{L}\rho_l||w^{(l)}||^2\\s.c.\space g(X)\ge0, h(X^{(0)})=0,c \in \Bbb{R}^L
$$




### 2.1.3 Lagrangien du problème

$L(w,X,\mu)=f(y,X^{(l)})-\lambda^T.g(w,X)-\mu.h(X^{(0)})$, 

avec$\space \lambda \ge0, \mu \in \Bbb{R^n} $

* Problème primal: $(P)$

  $$
  min \space l(y,\phi^{(L)}(w^{(L)}X^{(L)}+b^{(L)}))+ \sum\limits_{l=0}^{L-1} c_{l+1} || X^{(l+1)} - w^{(l)}\cdot X^{(l)}-b^{(l)}||^2 +\sum\limits_{l=0}^{L}\rho_l||w^{(l)}||^2\\s.c.\space g(X)\ge0 + , h(X^{(0)})=0
  $$

* Problème dual: $(D)$

$$
min \space l(y,\phi^{(L)}(w^{(L)}X^{(L)}+b^{(L)}))+ \sum\limits_{l=0}^{L-1} c_{l+1} || X^{(l+1)} - w^{(l)}\cdot X^{(l)}-b^{(l)}||^2 +\sum\limits_{l=0}^{L}\rho_l||w^{(l)}||^2\\s.c.\space g(X)\ge0 + , h(X^{(0)})=0 $$
$$

  


$$
  d(\mu,\lambda)=\mathop{Inf}\limits_{(w,X)} L(w,X,\mu) =\mathop{Inf}\limits_{(w,X)} \space\space f(y,X^{(l)})-\lambda^T.g(X)-\mu.h(X^{(0)})
$$




## 2.2 Qualification des constraintes

* $\vec{e_{w_i}}(\vec{e_{X_i}})$,vecteur correspond  à la paramètre $w_i(X_i)$. Remarquons que les vecteurs $\vec{e_{w_i}},\vec{e_{X_i}}$ forment une base de l’espace des paramètres.

* $I = A\Big(X\Big)$  ensemble des indices des contraintes actives

* $S = \{X\in \Bbb{R}^M:g(X)\ge0,h(X)=0\}  $ domaine des contraintes

* M nombre du $X$ dans tous les couches  $  \forall l \in [1,L]$

* N nombre du X paramètres dans couche 0

* $\vec{X}$ vecteur de tous les paramètres $X$ dans $(P)$

  $\vec{X}[0:N]$ correspond aux paramètres dans couche 0

### 2.2.1 Points réguliers du problème

Dans ce problème, le domaine de contraint $S$ est un plan dans l’espace $\Bbb{R}^M$, donc il est convexe et fermé.

Trouvons les points réguliers du problème:

Notons $g: \Bbb{R}^M \rightarrow \Bbb{R}^M$

$$
 g: \Bbb{R}^M \rightarrow \Bbb{R}^M\\
 X \rightarrow [X^{(l)}_{1};X^{(l)}_{2};...;X^{(L)}_1;...;X^{(L)}_{m}]
$$
Donc $g_i = X_i, X_i \in X^{(l)},l \in[0,L]$

* **si $w^*$ est un point intérieur à S**

​	Par définition,$X_i > 0,\forall X_i \in \vec{X}$, $ X_i = 0, \forall i \in [0,N] $

​	on a $g_i(w^*)= w_i^* > 0, \forall i \in [1,M] $, tous les contraint sont  inactive

​	Donc ,$L(w^*)= \{X|X_i = 0, \forall i \in [M+1,N]\} = \overline{Z(w^*)}$

* **Si $w^*$ est un point frontière à S**

Soit $I = A\Big(W^*\Big)$

$$
\mathop{w_i^*} = 0, \forall i \in I\space \textbf{et} \space w^*_i\ge0, \forall i \in [1,n]\setminus
$$

$$
g_i(w^*)=0, \forall i \in I, \space et \space g_i(w^*)>0, \forall i \in [1,n]\setminus I​$
$$



$\{g_i(w^*), for \space i \in I\}$ sont les seules contraints active. 

$\bigtriangledown g_i(w^*)\ge0, \forall i \in A\Big(w^*\Big); $

**La fermeture d’ensemble des directions réalisables:**
$$
\overline{Z(w^*)}=Vect\{(\vec{e_{w_i}})_{i \in [1,n]/I}\}
$$
Par définition, $L(\vec{w^*})= \{d\in\Bbb{R^n}: d^T.\Delta g_i(\vec{w^*})\ge0, \forall i \in A\Big(\vec{w^*}\Big); \}$ 

comme ici, $\bigtriangledown g_i(W^*) = \vec{ e_{w_i}}\forall i \in [0,M]$ $\bigtriangledown h_i(W^*) = \vec{ e_{w_i}}, \forall i \in [M+1,M+N]$

Alors,  $d \in L(w^*)$

$\iff \forall i \in A\Big(\vec{w^*}\Big), d^T.\big(\vec{e_{w_i}\big)}\ge 0 ; d^T \cdot \bigtriangledown h_j(w^*)=0, \forall j \in[1,N-M]$

$\iff \forall i \in A\Big(\vec{w^*}\Big),d_i\ge0,i \in [1,N]; d_i = 0, \forall i \in[N,M]$

**C’est exactement La fermeture d’ensemble des directions réalisables $\overline{Z(w^*)}$.**

En conclusion, on a montre que tous les $\vec{w} \in S$ sont les points réguliers, c’est aussi la qualification de contraintes.

### 2.2.2 Deuxième méthode pour la qualification de contraintes Independence Linéaire

 En effet, dans la démonstration précédente, nous avons vus que

$\bigtriangledown g_i(\vec{w^*}) = \vec{e_{i}}$, remarquer que $\forall i \in A\Big(w^*\Big), i \in [1,M] $

 Donc la famille 
$$
\{\bigtriangledown g_i(w^*),\forall i \in A\Big(w^*\Big) , \bigtriangledown h_i =  \vec{e_{i}},\forall i \in [M,N]\}
$$
est libre puisque la famille  $\{ \vec{e_{i}, \forall i \in A\Big(w^*\Big)}\}$ est libre.

Donc, la qualification de contraintes est vérifié. On peut appliquer les conditions de Karush-Kuhn-Tucker

## 2.3 Condition de Karush-Kuhn-Tucker

### 2.3.1 les conditions nécessaires pour que $\vec{w^*}$ soit un minimum local:

* réalisabilité primale 

  ​	$w^* \in S \iff w^*_i \ge 0,w^*_j=0, \forall i \in [1,M], j \in [M+1,M+N] $

* Réalisabilité duale et stationnarité

  rappellons  

  ​	$\exist \lambda^* \ge 0,\mu \in \Bbb{R}^N \bigtriangledown_{w,X}L(w^*,X^*,\lambda^*,\mu^*) = 0$

  * $\bigtriangledown_{w_i}f = Y_i\cdot\frac{X_i^{(L)}}{w_i^{(L)}\cdot X_i^{(L)}+b_i^{(L)}}\vec{e_{w_i}}+ 2\rho_l w_i^{(L)}\vec{e_{w_i}} , \forall w_i \in w^{(L)}$, 

  * $\bigtriangledown_{X_i}f = Y_i\cdot\frac{w_i^{(L)}}{w_i^{(L)}\cdot X_i^{(L)}+b_i^{(L)}}\vec{e_{X_i}}-2\lambda_{L}(X^{(L)}-w^{L-1}X^{(L-1)}-b^{(L-1)}), \forall X_i \in w^{(L)}$

  * $\bigtriangledown_{X^{(l)}}f = -2\lambda_{l+1} (X_i^{(l+1)}-w_i^{(l)}X_i^{(l)}-b_i^{(l)})\cdot X_i^{(l)}+ 2\lambda_l(X_i^{(l)}-w_i^{(l-1)}X^{(l-1)}-b_i^{(l-1)}), l \in [0,L-1]$

  * $\bigtriangledown_{w^{(l)}}f = -2\lambda_{l+1}(X^{(l+1)}-w^{l}X^{(l)}-b^{(l)})\cdot X^{(l)}+ 2\rho_l w_i^{(l)}, \forall l\in [0,L-1]$

  * $\bigtriangledown_{X_i^{(l)}}\lambda^Tg = \lambda_i\vec{e_{X_i}}, \forall X_i^{(l)} \in X_i^{(l)}, \forall l \in [0,L-1]$

  * $\bigtriangledown_{X_i}\mu^Th =\mu_i\vec{e_{X_i}} ,\forall X_i \in X^{(0)} $

  Donc,

  $$
  \bigtriangledown_{w,X}L(w^*,X^*,\lambda^*,\mu^*) = 0
  $$
  $  \iff\\$

  * $\bigtriangledown_{X^{(l)}}f (w^*,X^*,\lambda^*,\mu^*)= -\lambda^*, \bigtriangledown_{W^{(l)}}f (w^*,X^*,\lambda^*,\mu^*)= 0,l \in [1,L-1]$)

  * $\bigtriangledown_{W^{(L)}}f(w^*,X^*,\lambda^*,\mu^*)= \bigtriangledown_{X^{(L)}}f (w^*,X^*,\lambda^*,\mu^*)=0,\forall X_i \in X^{(L)} $    

  * $\bigtriangledown_{W_i}f(w^*,X^*,\lambda^*,\mu^*)= -\mu_i^*, \forall X_i \in X^{(0)}$    

  

* complémentarité

  ​	$(g(X^*))^T.\lambda^* = 0 \iff (X_i)^*.(\lambda_i)^* = 0, \forall X_i \in X^{(l)},\lambda_i \in\lambda^{(l)},l \in[0,L]$

Nous avons vu que les problèmes du $(P)$ est très difficile à résoudre et les conditions du optimisation, ne sont pas aussi pas facile à vérifier à cause de la dimension haute de la espace des variables $W,X$. Le problème transformée est plus difficile que le problème original à ce sens. C’est pourquoi on introduit BCD dans la partie suivante pour décomposer $(P)$ en $L$ sous-problème convexe.

Si on considère le problème couche par couche, c’est à dire fixer tous les paramètres sauf $w^{(l)},X^{(l)},b^{(l)}$, on peut décomposer $(P)$ comme un suite des problèmes $(P_l)$ définie comme:

* min $$\space \sum\limits_{-X_i \in X^{(L+1)}}Y_i\cdot ln(X_i)+c_{L+1} || X^{(L+1)} - w^{(L)}\cdot X^{(L)}-b^{(L)}||^2 \space \space\space (P_L)\\s.c.\space X^{(L)}\ge0, X^{(L+1)}\cdot \textbf{1} = 1$$

* min $$\space c_{l+1} || X^{(l+1)} - w^{(l)}\cdot X^{(l)}-b^{(l)}||^2 +\rho_l||w^{(l)}||^2\space \space\space (P_l)\\s.c.\space X^{(l)}\ge0$$

​        $ \forall l \in [1,L-1]  $

* min $$\space c_{0} || X^{(1)} - w^{(0)}\cdot X^{(0)}-b^{(0)}||^2 +\rho_l||w^{(0)}||^2\space \space\space (P_0)\\s.c.\space X^{(0)}\ge0,X^{(0)}-X =0$$

Tous les sous-problèmes $P_l$sont les problèmes de l’optimisation convexe (quadratique) avec les seules variables $w^{(l)},X^{(l)},b^{(l)}$.

### 2.3.2 Initialisation du résolution 

On va le discuter après la présentation de BCD pour voir mieux (voir **la partie 3.1.3**)



# 3 Analyse algorithmique

## 3.1 Algorithme BCD

BCD: Block-Coordinate Descent, c’est un algorithme qui permet de résoudre le modèle d’optimisation. Cet algorithme peut considérer en deux parties: renouveler les variables $ W $ et $b$ puis renouveler $X$.

### 3.1.1 Renouveler $ W _l$ et $b_l$ pour couche $l$

Pour certaine couche $l = 0,...,L$, on fixe tout d’abord les variables artificiels $X$, on renouvelle les paramètres $ W _l$ et  $b_l$ pour couche $l$ de 0 à $L$, on a les nouveaux variables $( W ^+_l,b^+_l)$ sous la forme:
$$
\left( W _{l}^{+}, b_{l}^{+}\right)=\arg \min _{ W , b} c_{l+1} D_{l}\left( W  X_{l}+b \mathbf{1}^{T}, X_{l+1}\right)+\pi_{l}( W )
$$
où $D_l$ est une fonction dépend de l’entrée, de la sortir et de la fonction d’activation d’une couche, par exemple si la fonction d’activation de couche $l$ est ReLu, on a donc d’apres (15):
$$
\left( W _{l}^{+}, b_{l}^{+}\right)=\arg \min _{ W , b} c_{l+1}\left\| W  X_{l}+b \mathbf{1}^{T}-X_{l+1}\right\|_{F}^{2}+\rho_{l}\| W \|_{F}^{2}
$$

### 3.1.2 Renouveler $X$ pour couche $l$

De même, pour $l=0,...,L$, on a le $X^+$ renouvelé:
$$
\begin{array}{c}
X_{l}^{+}=\arg \min _{Z} c_{l+1} D_{l}\left( W _{l} Z+b_{l} \mathbf{1}^{T}, X_{l+1}\right) \\
+c_{l} D_{l-1}\left(Z, X_{l-1}^{0}\right)
\end{array}
$$
où $X_{l-1}^{0}:= W _{l-1} X_{l-1}+b_{l-1} \mathbf{1}^{T}$ , ce probleme est un problème convexe.

On se concentre maintenant sur un cohue $l$ qui porte une fonction d’activation ReLu:
$$
\begin{aligned}
X_{l}^{+}=\arg \min _{Z \geq 0} c_{l+1}\left\|X_{l+1}- W _{l} Z-b_{l} 1^{T}\right\|_{}^{2}+\\
c_{l}\left\|Z- W _{l-1} X_{l-1}-b_{l-1} 1^{T}\right\|_{}^{2}
\end{aligned}
$$
Mais pour la dernière couche la situation est differente, parce qu’il faut penser à la fonction de perte, donc le $X_{L+1}$ renouvelé est:
$$
X_{L+1}^{+}=\arg \min _{Z} l(Y, Z)+c_{L+1} D_{L}\left(X_{L}^{0}, Z\right)
$$
avec  $X_{L}^{0}:=W_{L} X_{L}+b_{L} 1^{T}$.



## 3.2 Réalisation 

On détaille l'algorithme BCD dans notre code.

On redit notre modèle utilisé dans le code. Ce modèle est équivalent au modèle utilisé dans l'analyse précédente. Pour un réseau neuronal, la fonction objective est:
$$
\min_{W,b} \mathcal L(Y,\Phi_{W,b}(X))
$$
où
$$
\Phi_{W,b}(X) = \sigma^L(b^L+W^L\sigma^{L-1}(b^{L-1}+W^{L-1}\sigma^{L-2}(...)))
$$
$ \mathcal L $ est une fonction de perte, $ X $ est l'entrée, $ Y $ est l'étiquette, L est le nombre de couches, $ W ^ L $ est le poids de la L-ème couche, $ b^L $ est la le biais de la L-ème couche, et $ \sigma ^ L $ est la fonction d'activation de la L-ème couche.

Le problème est équivalent à
$$
\min_{W,b} \mathcal L(Y,X^{L+1})\\
\text{s.c. } \forall i \in \left\{1,2,...,L \right\}, X^{L+1} = \sigma^L(W^L X^L + b^L)
$$
Nous définissons une nouvelle fonction objectif en découplant les contraintes:
$$
\min_{W,b,X} \mathcal L(Y,X_{L+1}) + \sum_{i=1}^L \lVert X^{L+1} - \sigma^{L}(W^LX^L+b^L) \rVert_2^2
$$
Ensuite, nous pouvons minimiser W, b, X dans chaque couche. L'algorithme BCD est:

-----

***Pseudo-code de L'algorithme BCD***

$X^{L+1} \leftarrow \arg\min \mathcal L(Y,X_{L+1}) +\lVert X^{L+1} - \sigma^{L}(W^LX^L+b^L) \rVert_2^2$

**For** i in {L, L-1, ... , 1}:

​    $W^i \leftarrow \arg \min \lVert X^{i+1} - \sigma^{i}(W^iX^i+b^i) \rVert_2^2$

​    $b^i \leftarrow \arg \min \lVert X^{i+1} - \sigma^{i}(W^iX^i+b^i) \rVert_2^2$

​    $X^i \leftarrow \arg \min \lVert X^{i+1} - \sigma^{i}(W^iX^i+b^i) \rVert_2^2 + \lVert X^{i} - \sigma^{i}(W^{i-1}X^{i-1}+b^{i-1}) \rVert_2^2$

End **For**

-----

Malheureusement, nous n'avons trouvé que quelques articles convexes, sur lesquelles nous avons avancé quelques idées. Ces idée sont codées respectivement `optimizers.BCD` et `optimizers.BCD_V2`.

### 3.2.1 Idée 1

Pour renouveler $X^{L+1}$, on simplement imposons que  $X^{L+1} \leftarrow Y$. Cela permet à la vérité fondamentale d'être efficacement réinjectée dans le réseau neuronal.

Pour renouveler les autres, on utilise la descente de gradient.

On introduit comment calculer les gradients dans la suite.

Posons $f(W^i,b^i) = \lVert X^{i+1} - \sigma^{i}(W^iX^i+b^i) \rVert_2^2$ , alors:
$$
\begin{aligned}
\frac{\partial f}{\partial b^i} &= \frac{\partial f}{\partial \sigma^i} \cdot \frac{\partial \sigma^i}{\partial b^i}\\
&= \nabla\sigma^i . (-2X^{i+1}+2\sigma^i)
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial f}{\partial W^i} &= \frac{\partial f}{\partial \sigma^i} \cdot \frac{\partial \sigma^i}{\partial W^i}\\
&=\nabla\sigma^i .(-2X^{i+1}+2\sigma^i) . {X^i}^T
\end{aligned}
$$

Posons
$$
\begin{aligned}
g(X^i) &= \lVert X^{i+1} - \sigma^{i}(W^iX^i+b^i) \rVert_2^2 + \lVert X^{i} - \sigma^{i}(W^{i-1}X^{i-1}+b^{i-1}) \rVert_2^2\\
&\overset{\Delta}{=} g_1(X^i)+g_2(X^i)
\end{aligned}
$$
Alors
$$
\begin{aligned}
\frac{\partial g_1}{\partial X^i} &= \frac{\partial g_1}{\partial \sigma^i} \cdot \frac{\partial \sigma^i}{\partial X^i}\\
&= {W^i}^T . \nabla \sigma^i . (-2X^{i+1}+2\sigma^i)
\end{aligned}
$$
De même pour $\frac{\partial g_2}{\partial X^i}$ .

### 3.2.2 Idée 2

Dans l'idée 1, $X^{L+1} \leftarrow Y$ ne correspond pas $X^{L+1} \leftarrow \arg\min \mathcal L(Y,X_{L+1}) +\lVert X^{L+1} - \sigma^{L}(W^LX^L+b^L) \rVert_2^2$ . Donc nous essayons de trouver un autre méthode.

**Initialisation du résolution** 
$$
\space \sum\limits_{X_i \in X^{(L+1)}}-Y_i\cdot ln(X_i)+c_{L+1} || X^{(L+1)} - W^{(L)}\cdot X^{(L)}-b^{(L)}||^2 \space \space\space (P_L)\\s.c.\space X^{(L)}\ge0,X^{(L+1)}\cdot \textbf{1} = 1
$$

posons $p$ le nombre des paramètres du $X^{(L+1)}$, c’est donc le nombre du classification.

$ X^{0,L}=W^{(L)}\cdot X^{(L)}+b^{(L)}$

**Lagrangien du problème $P^{(L)}$ :**

$$ L(X^{(L+1)},\mu) = \sum\limits_{i=0}^p -Y_iln(X_i^{(L+1)})+\sum\limits_{i=0}^p C_{L+1}|X_i^{(L+1)}-X_i^{0,L}|^2-\mu(X^{L,0}\cdot \textbf {1}-1)$$

$\iff$

$$ L(X^{(L+1)},\mu) =\mu+ \sum\limits_{i=0}^p\Big(-Y_iln(X_i^{(L+1)})+C_{L+1}(X^{(L+1)}_i)^2-X^{(L+1)}_i(\mu+2C_{L+1}X_i^{0,L})\Big)$$

**qualification des contraintes** :

Comme $\forall X^{(L+1)} \in S,\{\bigtriangledown (X^{(L+1)}\cdot \textbf{1} -1)\}$ est libre , d’après $QC-IL$ les qualification des contraintes sont vérifiés. 

**les conditions de KKT**:

Comme $(P_L)$est une optimisation convexe, les conditions de KKT est nécessaire et suffisante pour trouver la solution optimale:
$$
\frac {\partial L(X^{(L+1)},\mu)}{X_i^{(L+1)}}=\frac{-Y_i}{X_i^{L+1}} + 2C_{L+1}\cdot(X_i^{L+1})-(\mu +2C_{L+1}X_i^{0,L}) = 0
$$

$$
\iff(X_i^{(L+1)})^2 +\frac {\mu+2C_{L+1}X_i^{0,L}}{2C_{L+1}}X_i^{(L+1)} -\frac{Y_i}{2C_{L+1}}=0
$$

En prenant la racine non négative:

$$
X_i^{(L+1)} = \frac{1}{2}\Big(\frac{\mu}{2C_{L+1}}+X_i^{0,L}+\sqrt{(\frac{\mu}{2C_{L+1}}+X_i^{0,L})^2+\frac{2Y_i}{C_{L+1}}}\Big)
$$
On peut trouver une solution explicite du $X^{(L+1)}$ par $\mu$ dans la théorie et en pratique on peut utiliser la méthode de dichotomie pour le calculer. 

**méthode de dichotomie**

* La borne supérieure:

  comme $X_i^{*(L+1)} \le 1 \Rightarrow \frac{\mu}{2C_{L+1}}+X_i^{0,L} <  1 $

  donc

$$
\mu <2C_{L+1} \big(1- \mathop{max}\limits_{0\le i \le p}(X_i^{0,L})\big)
$$

* La borne inférieure:

  Soit 

  $Z$ une solution quelconque du problème $P_L$, $Z \ge 0, \textbf{1}^T\cdot Z =1$

  $X^{*(L+1)}$ la solution optimale

  

  . Nous avons:
  $$
  \forall i \in [0,p],\space-ln(X_i^{(L+1)}) \le -\sum\limits_{i=0}^p Y_iln(X_i^{*(L+1)}) \le p^* \le \theta\\
  avec,\space \theta := - \sum\limits_{p=0}^p Y_iln(Z_i^{*(L+1)})
  $$
  donc $X_i \ge e^{-\theta}, \forall i \in [0,p]$ avec la condition d’optimisation $(36)$ on a:

  $X_i^{*(L+1)} = \frac{1}{2}\Big(\frac{\mu}{2C_{L+1}}+X_i^{0,L}+\sqrt{(\frac{\mu}{2C_{L+1}}+X_i^{0,L})^2+\frac{2Y_i}{C_{L+1}}}\Big)\ge e^{-\theta}$

  $\Rightarrow 2(\frac{\mu}{2C_{L+1}}+X_i^{0,L})+\sqrt{\frac{2Y_i}{C_{L+1}}}\ge e^{-\theta}$
  $$
  \iff \mu \ge C_{L+1}\Big(e^{-\theta}-\mathop{Min}_{0\le i\le p}\big(2X_i^{(0,L)}+\sqrt{\frac{2Y_i}{C_{L+1}}}\big)\Big)
  $$

* Conclusion la borne de $\mu$ pour la méthode de dichotomie
  $$
  C_{L+1}\Big(e^{-\theta}-\mathop{Min}_{0\le i\le p}\big(2X_i^{(0,L)}+\sqrt{\frac{2Y_i}{C_{L+1}}}\big)\Big) \le \mu <2C_{L+1} \big(1- \mathop{max}\limits_{0\le i \le p}(X_i^{0,L})\big)
  $$

<div style="page-break-after: always; break-after: page;"></div>

## 3.3 Expérience numérique

Le modèle décrit dans cet article a été testé sur l'ensemble de données MNIST (LeCun & Cortes, 2010). Pour le problème de classiﬁcation, l'ensemble de données a été divisé en 60 000 échantillons d'apprentissage et 10 000 échantillons d'essai avec une entropie croisée de softmax.

On a fait les expereinces avec seulement les couches linéaires pour gagner du temps, comme une couche de convolution est équivalent à une couche linéaire(Ma, W., & Lu, J. 2017. An equivalence of fully connected layer and convolutional layer.).

On utilisé dans l’ensemble 3 méthodes pour faire la classification, voici les résultats:

![SGD](D:%5C%E4%B8%AA%E4%BA%BA%E9%A1%B9%E7%9B%AE%5C%E6%9C%80%E4%BC%98%E5%8C%96%E8%AF%BE%E9%A2%98%5CMod%C3%A9lisation.assets%5CSGD.png)

<center>Figure 5 - SGD</center>

![BCD](D:%5C%E4%B8%AA%E4%BA%BA%E9%A1%B9%E7%9B%AE%5C%E6%9C%80%E4%BC%98%E5%8C%96%E8%AF%BE%E9%A2%98%5CMod%C3%A9lisation.assets%5CBCD.png)

<center>Figure 6 - BCD idée 1</center>

![BCD2](D:%5C%E4%B8%AA%E4%BA%BA%E9%A1%B9%E7%9B%AE%5C%E6%9C%80%E4%BC%98%E5%8C%96%E8%AF%BE%E9%A2%98%5CMod%C3%A9lisation.assets%5CBCD2.png)

<center>Figure 7 - BCD idée 2</center>

Le temps moyen de parcourir un période correspond de ces trois méthodes sont :

<div style="page-break-after: always; break-after: page;"></div>

**Table  1 - temps pour un période**

| Méthode | Temps moyen(s) |
| :-----: | :------------: |
|   SGD   |      7.96      |
|  BCD 1  |      5.75      |
|  BCD 2  |      21.8      |

On voit bien que la méthode 1 de BCD  est beaucoup meilleure que la méthode 2, par rapport à  la méthode de SGD, elles ont presque la même précision mais SGD est un peu meilleure, mais quant au complexité, d’après le temps moyen de parcourir un période, c’est BCD 1 qui est meilleure.

De plus, notre code de l’algorithme de BCD supporte le fonctionnement de mini-batch qui permet de combiner les avantages de SGD et de BCD.

