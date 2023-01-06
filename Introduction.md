

# The $L_{2,d}$-spectral radius dimensionality reduction.
Here we apply the notion of the $L_{2,d}$-spectral radius dimensionality reduction to natural language processing, graphs and possibly other areas of machine learning. 

Suppose that $(A_1,\dots,A_r)$ are $n\times n$-complex matrices. Then define the $L_{2,d}$-spectral radius of $(A_1,\dots,A_r)$ denoted $\rho_{2,d}(A_1,\dots,A_r)$ to be the maximum value of $\frac{\rho(A_1\otimes X_1+\dots+A_r\otimes X_r)}{\rho(X_1\otimes\overline{X_1}+\dots+X_r\otimes\overline{X_r})}$. One should consider the $L_{2,d}$-spectral radius as a generalization of the notion of the spectral radius of a complex matrix or operator to multiple matrices or operators.

We say that a tuple $(X_1,\dots,X_r)$ is a $L_{2,d}$-spectral radius dimensionality reduction (LSRDR) of $(A_1,\dots,A_r)$ if $\frac{\rho(A_1\otimes\overline{X_1}+\dots+A_r\otimes\overline{X_r})}{\rho(X_1\otimes\overline{X_1}+\dots+X_r\otimes\overline{X_r})}=\rho_{2,d}(A_1,\dots,A_r)$.






**Possible advantages**

1. LSRDRs seem to have a decent (but mostly undeveloped) mathematical theory behind them that appeals to pure mathematicians. Machine learning algorithms that have a mathematical theory behind them should be more explainable and interpretable than less mathematical machine learning models. A mathematical theory behind LSRDRs could also mean that LSRDRs can be applied to other disciplines besides machine learning including quantum information theory and quantum mechanics.

2. (global convergence to unique optimum) The $L_{2,d}$-SRDR $(X_1,\dots,X_r)$ of $(A_1,\dots,A_r)$ is usually essentially unique. Furthermore, gradient ascent-like algorithms will (with few exceptional circumstances) converge to the global maximum and not just a local but non-global maximum. This essential uniqueness will aid with interpretability and explainability; it is more difficult to interpret or explain a model if result of the trained model depends on what local optimum the gradient ascent algorithm has reached since the choice of local optimum is some random information and such random information impedes interpretability/explainability.

3. For word and graph embeddings, tokens and graphs are often represented by matrices and not just vectors. Matrices have much richer structure than vectors, and this structure can be useful in NLP and analyzing data represented as graphs. For example, in NLP, words often have multiple meanings. When one represents words as vectors, the vectors may perform well at capturing a single meaning of a word, but in order to capture multiple meanings of a word simultaneously, it is best to use multiple vectors, but matrices may be better at capturing multiple meanings of words since matrices consist of multiple vectors. In fact, in NLP when we associate tokens with matrices instead of simply vectors, these matrices can produce a contextual embedding of words without needing to use any neural networks. Since matrices can be multiplied together, one can associate sentences with products of matrices. By associating sentences with products of matrices, a computer can tell whether such a sentence is grammatically correct, and a computer can even generate new sentences just by using the word embedding and algorithms like simulated annealing and without using any neural networks.

**Possible disadvantages**

1. The spectral radii approach is mostly untested and not much is known about it.

2. While LSRDRs are useful for a couple of niches such as analyzing simple block cipher round functions and constructing word embeddings along with graph embeddings, it is unclear how one may use LSRDRs for other tasks in machine learning. For example, it is unclear how LSRDRs may be useful for classifying images or if LSRDRs can perform as well as neural networks for classification problems.

3. It is unclear as to how well LSRDRs can compete with other approaches. The $L_{2,d}$-spectral radius is necessary for measuring the security of simple block ciphers such as Circcash's Hashspin mining algorithm. On the other hand, there are plenty of graph embedding and word embedding algorithms to choose from already. It is unclear what advantage LSRDRs will have over the current state of the art.

4. When using LSRDRs to build more complicated models, one will probably need to use both LSRDRs and neural networks, but it is not clear whether neural networks will interact with LSRDRs very well.

**Business inquiries**

If you are considering using LSRDRs for your project or business but don't know where to begin, you may send a message to circcash9192020 (AT) protonmail.com. Right now, the Circcash developer is the only entity who is attempting to use spectral radii for machine learning, so you will not find anyone else who can replicate these services.
