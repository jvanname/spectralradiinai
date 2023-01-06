# spectralradiiai
Here we apply the notion of the $L_{2,d}$-spectral radius dimensionality reduction to natural language processing, graphs and possibly other areas of machine learning. 

Suppose that $(A_1,\dots,A_r)$ are $n\times n$-complex matrices. Then define the L_{2,d}-spectral radius of $(A_1,\dots,A_r)$ denoted $\rho_{2,d}(A_1,\dots,A_r)$ to be the maximum value of $\frac{\rho(A_1\otimes X_1+\dots+A_r\otimes X_r)}{\rho(X_1\otimes\overline{X_1}+\dots+X_r\otimes\overline{X_r})}$. 

We say that a tuple $(X_1,\dots,X_r)$ is a $L_{2,d}$-spectral radius dimensionality reduction (LSRDR) of $(A_1,\dots,A_r)$ if $\frac{\rho(A_1\otimes\overline{X_1}+\dots+A_r\otimes\overline{X_r})}{\rho(X_1\otimes\overline{X_1}+\dots+X_r\otimes\overline{X_r})}=\rho_{2,d}(A_1,\dots,A_r)$.






**Possible advantages**

1. LSRDRs seem to have a decent (but mostly undeveloped) mathematical theory behind them that appeals to pure mathematicians. Machine learning algorithms that have a mathematical theory behind them should be more explainable and interpretable than less mathematical machine learning models. A mathematical theory behind LSRDRs could also mean that LSRDRs can be applied to other disciplines besides machine learning including quantum information theory and quantum mechanics.

2. The $L_{2,d}$-SRDR $(X_1,\dots,X_r)$ of $(A_1,\dots,A_r)$ is essentially unique. Furthermore, gradient ascent-like algorithms will (with few exceptional circumstances) converge to the global maximum and not just a local but non-global maximum. This essential uniqueness will aid in the interpretability and explainability of models using LSRDRs since 

3. For word and graph embeddings, tokens and words are represented by matrices and not just vectors. Matrices have much richer structure than vectors, and matrices are useful in 


**Possible disadvantages**

1. The spectral radii approach is mostly untested and not much is known about it.

2. While LSRDRs are useful for a couple of niches such as analyzing simple block cipher round functions and constructing word embeddings along with graph embeddings, it is unclear how one may use LSRDRs for other tasks in machine learning. For example, it is unclear how LSRDRs may be useful for classifying images or if LSRDRs can perform as well as neural networks for classification problems.

3. It is unclear as to how well LSRDRs can compete with other approaches. The $L_{2,d}$-spectral radius is necessary for measuring the security of simple block ciphers such as Circcash's Hashspin mining algorithm. On the other hand, there are plenty of graph embedding and word embedding algorithms to choose from already

4. 
