Let $V$ be a set, and let $w:P(V)\setminus\{\emptyset\}\rightarrow[0,1]$ be a function with $\sum_{R\subseteq V,R\neq\emptyset}w(R)=1$. Suppose that $m:V\rightarrow\[0,\infty)$ is a function. Then define $L(f,m,w)=\sum_{R\subseteq V,R\neq\emptyset}w(R)\cdot\log(\\|\sum_{v\in R}f(v)\\|)-\log(\\|\sum_{v\in V}m(v)f(v)f(v)^\ast\\|)/2$.
Let $K$ denote either the field of real or complex numbers. Let $d$ be a natural number. We say that a function $f:V\rightarrow K^d$ is a geometric mean of norm optimized (GMNO) hypergraph pre-embedding of the weighted hypergraph $w$ with multiplicity function $m$ if the quantity $L(f,m,w)$ is locally maximized. We say that a GMNO hypergraph pre-embedding $f$ is a GMNO hypergraph embedding if the matrix $\sum_{v\in V}m(v)m(v)f(v)^\ast$ is a diagonal matrix with increasing diagonal entries. Of $v_0\in V$, then we say that a GMNO hypergraph embedding $f$ is positive for $v_0$ if each entry in $f(v_0)$ is positive.

**Desirable properties**

Uniqueness: The GMNO hypergraph embedding $f$ for multiplicity function $m$ that is positive for $v_0$ with $\\|f(v_0)\\|=1$ is typically unique in the sense that if $f:V\rightarrow K_1^d,g:V\rightarrow K_2^d$ are two GMNO hypergraph embeddings with multiplicity functions $m$ that are positive for $v_0$ with $\\|f(v_0)\\|=\\|g(v_0)\\|=1$ and which have been trained with gradient descent but which have different initial conditions and possibly different optimizers, then we would typically have $f=g$. If $K_1=\mathbb{C},K_2=\mathbb{R}$, then by uniqueness, the GMNO hypergraph embedding $f$ will automatically be a real-valued function.

Smoothness: The GMNO hypergraph embedding $f$ for multiplicity function $m$ that is positive for $v_0$ with $\\|f(v_0)\\|=1$ changes moderately one changes the hypergraph $f$ slightly. Furthermore, the Gram matrix of $f$ will change even less since the process of selecting a canonical basis of $f$ is discontinuous. The quantity $L(f,m,w)$ will change even less as one modifies the hypergraph $f$.

Sphericity: If $f$ is a GMNO hypergraph embedding for a suitable multiplicity function $m$, then $\|f(v)\|$ will typically have very low variance. Therefore, the set $f[V]$ will be near the sphere $S^{d-1}$ as long as $E(\|f(v)\|)=1$ even though the fitness function for $f$ does not explicitly require for $f[V]$ to be near the sphere $S^{d-1}$.

Thin singularities: While the function $f\mapsto L(f,m,w)$ does have singularities, the singularities of this function are of a small dimension and quite thin due the logarithmic function. The singularities of this function therefore do not disrupt the usability of GMNO hypergraph embeddings.

**Other properties**

Uncentered: GMNO hypergraph embeddings $f$ typically do not have mean near $0$.

Dimension filling: In order for a GMNO hypergraph embedding to minimize $\\|\sum_{v\in V}m(v)f(v)f(v)^\ast\\|$, the matrix $\sum_{v\in V}m(v)f(v)f(v)^\ast$ will sometimes be near a constant multiple of the identity matrix. In other words, the function $f$ will want to use all of the available dimensions nearly equally. This may result in the GMNO hypergraph embedding using dimensions when it is not necessary for hypergraph embeddings to use such dimensions.

Flat embedding: Sometimes the matrix $\sum_{v\in V}m(v)f(v)f(v)^\ast$ can be approximated by a low rank matrix. This means that not all the dimensions in $K^d$ are adequately used.

Non-repulsion: For GMNO hypergraph embeddings, it is feasible that there are different nodes $u,v$ where $f(u)$ and $f(v)$ are extremely close together since GMNO hypergraph embeddings repel nodes away from each other only by requiring all dimensions to be used nearly equally. Of course, this characteristic can be ameliorated by increasing the dimension $d$ of the vector space $K^d$.

**Applications**

Positional embeddings: GMNO hypergraph embeddings may be used in convolutional neural networks to encode location. Let $V=\\{1,\dots, m\\}\times\\{1,\dots,n\\}$. Let $E=\\{(i,j),(i+1,j),(i,j+1),(i+1,j+1)\mid 1\leq i<m,1\leq j<n\\}.$ Then a GMNO hypergraph embedding $f:V\rightarrow K^d$ maps a position $(i,j)$ to information $f(i,j)$ about the position $(i,j)$. It is less clear how one may use GMNO hypergraph embeddings as positional embeddings for transformers in natural language processing. While each dimension in GMNO hypergraph embeddings for the graph $(\\{1,\dots,n\\},\\{\\{i,i+1\\}\mid 1\leq i<n\\})$ resemble a sinusoid, the frequencies do not form a geometric distribution.

Recommender systems: 



