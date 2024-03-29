

# The $L_{2,d}$-spectral radius dimensionality reduction.
Here we apply the notion of the $L_{2,d}$-spectral radius dimensionality reduction to natural language processing, graphs and possibly other areas of machine learning. 

Suppose that $(A_1,\dots,A_r)$ are $n\times n$-complex matrices. Then define the $L_{2,d}$-spectral radius of $(A_1,\dots,A_r)$ denoted $\rho_{2,d}(A_1,\dots,A_r)$ to be the maximum value of $\frac{\rho(A_1\otimes X_1+\dots+A_r\otimes X_r)}{\rho(X_1\otimes\overline{X_1}+\dots+X_r\otimes\overline{X_r})}$. One should consider the $L_{2,d}$-spectral radius as a generalization of the notion of the spectral radius of a complex matrix or operator to multiple matrices or operators.

We say that a tuple $(X_1,\dots,X_r)\in M_d(\mathbb{C})^r$ is an $L_{2,d}$-spectral radius dimensionality reduction (LSRDR) of $(A_1,\dots,A_r)$ if $\frac{\rho(A_1\otimes\overline{X_1}+\dots+A_r\otimes\overline{X_r})}{\rho(X_1\otimes\overline{X_1}+\dots+X_r\otimes\overline{X_r})}$ is locally maximized. We say that an LSRDR $(X_1,\dots,X_r)$ of $(A_1,\dots,A_r)$ is optimal if $\frac{\rho(A_1\otimes\overline{X_1}+\dots+A_r\otimes\overline{X_r})}{\rho(X_1\otimes\overline{X_1}+\dots+X_r\otimes\overline{X_r})}=\rho_{2,d}(A_1,\dots,A_r)$. Computer experiments show that if $(X_1,\dots,X_r)$ is an optimal LSRDR of $(A_1,\dots,A_r)$ and $(X_1,\dots,X_r)$ have no common invariant subspace, then there are matrices $R,S$ where $A_j=RX_jS$ for $1\leq j\leq r$.

**Possible advantages**

1. LSRDRs seem to have a decent (but mostly undeveloped) mathematical theory behind them that appeals to pure mathematicians. Machine learning algorithms that have a mathematical theory behind them should be more explainable and interpretable than less mathematical machine learning models. A mathematical theory behind LSRDRs could also mean that LSRDRs can be applied to other disciplines besides machine learning including quantum information theory and quantum mechanics.

2. (Empirical confluence) Gradient ascent algorithms that calculate LSRDRs of $(A_1,\dots,A_r)$ will often converge to essentially the same LSRDR which is presumed to be optimal. This confluence property will aid with interpretability and explainability; it is more difficult to interpret or explain a model if result of the trained model depends on what local optimum the gradient ascent algorithm has reached since the choice of local optimum is some random information and such random information impedes interpretability/explainability. 

3. For word and graph embeddings, tokens and graphs are often represented by matrices and not just vectors. Matrices have much richer structure than vectors, and this structure can be useful in NLP and analyzing data represented as graphs. For example, in NLP, words often have multiple meanings. When one represents words as vectors, the vectors may perform well at capturing a single meaning of a word, but in order to capture multiple meanings of a word simultaneously, it is best to use multiple vectors, but matrices may be better at capturing multiple meanings of words since matrices consist of multiple vectors. In fact, in NLP when we associate tokens with matrices instead of simply vectors, these matrices can produce a contextual embedding of words without needing to use any neural networks. Since matrices can be multiplied together, one can associate sentences with products of matrices. By associating sentences with products of matrices, a computer can tell whether such a sentence is grammatically correct, and a computer can even generate new sentences just by using the word embedding and algorithms like simulated annealing and without using any neural networks.

4. (low variation) If $(A_1,\dots,A_r),(B_1,\dots,B_r)$ are nearby, then one can find $L_{2,d}$-SRDRs of $(A_1,\dots,A_r),(B_1,\dots,B_r)$ which are also nearby.

5. (Fast gradient computation) The gradient of any particular non-repeating eigenvalue of a matrix can be computed using the corresponding left and right eigenvectors [1]. The dominant left and right eigenvectors can easily be approximated and updated each round using a power iteration and by using matrices over the field of complex numbers (since real matrices do not necessarily have dominant eigenvalues).

6. (Symmetry preservation) The LSRDR of $(A_1,\dots,A_r)$ will inherit many of the properties (such as positive definiteness, Hermitianness, realness, and symmetry) of $(A_1,\dots,A_r)$.

**Possible disadvantages**

1. The spectral radii approach is mostly untested and not much is known about it.

2. While LSRDRs and similar algorithms are useful for a couple of niches such as analyzing simple block cipher round functions and constructing word embeddings along with graph embeddings, it is unclear how one may use LSRDRs for other tasks in machine learning. During training LSRDRs exhibit complicated behavior, but most of the complexity of $(X_1,\dots,X_r)$ is explicitly in the matrices themselves rather than in a function obtained from $(X_1,\dots,X_r)$. This makes LSRDRs and similar algorithms good for constructing word embeddings and graph embeddings, but it is unclear how LSRDRs can be used for other purposes.

3. It is not yet clear as to how well LSRDRs can compete with other approaches. The $L_{2,d}$-spectral radius is necessary for measuring the security of simple block ciphers such as Circcash's Hashspin mining algorithm. On the other hand, there are plenty of graph embedding and word embedding algorithms to choose from already. It is unclear what advantage LSRDRs will have over the current state of the art.

4. It is unclear as to whether LSRDRs will retain their desirable properties when one is using neural networks to compute very large LSRDRs.

**Business inquiries**

If you are considering using LSRDRs for your project or business but don't know where to begin, or you may send a message to circcash9192020 (AT) protonmail.com. Right now, the Circcash developer is the only entity who is attempting to use spectral radii for machine learning, so you will not find anyone else who can replicate these services. The Circcash developer is especially interested in applying LSRDRs to data that can be represented as a graph including social networks and recommender systems.

References:


[1] Econometric Theory, 1, 1985, 179-191. Printed in the United States of Amercia. ON DIFFERENTIATING EIGENVALUES AND EIGENVECTORS
