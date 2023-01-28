In this post, we shall formulate a mathematical description of a word embedding called a matrix product optimized word embedding (MPO) that maps tokens to matrices for NLP applications. MPO word embeddings are produced by gradient descent but they require no neural networks.

Since spaces of matrices have richer structure than just vector spaces or inner product spaces, MPO word embedddings can do things that other kinds of word embeddings are not capable of doing. MPO word embeddings can be used to produce contextual embeddings (such a contextual embedding requires no neural networks) for NLP applications; MPO word embeddings can distinguish between several meanings of a word. MPO word embeddings can also evaluate the local grammatical correctness of sentence. As a consequence, MPO word embeddings together with simulated annealing can be used to produce locally grammatically correct sentences without needing any neural networks. MPO word embeddings are globally convergent in the sense that if one trains a MPO word embedding twice with different intial conditions but with the same matrix embedding potential and corpus, the two word embeddings will be isomorphic and even equal after choosing a canonical basis and writing the matrices in terms of that basis. In fact, the word embeddings will be real-valued functions (after choosing a canonical basis) regardless of whether they have been trained using the field of comple numbers or not.

In this post, $K$ shall denote either the field of real or the field of complex numbers.

Suppose that $A$ is a set which shall be called the set of tokens.

Let $a_1\dots a_n$ be a string over the set $A$. The string $a_1\dots a_n$ shall be called the corpus.

A function $N:M_d(K)^r\rightarrow [0,\infty)$ shall be called a matrix embedding potential if

1. $N(A_1,\dots,A_r)=N(A_{\sigma(1)},\dots,A_{\sigma(r)})$ for all permutations $\sigma:\\{1,\dots,r\\}\rightarrow\\{1,\dots,r\\}$,

2. $N(\lambda A_1,\dots,\lambda A_r)=|\lambda|\cdot N(A_1,\dots,A_r)$ for each $\lambda\in K$, and

3. $N(UA_rU^{\ast},\dots,UA_rU^{\ast})=N(A_1,\dots,A_r)$ whenever $K=\mathbb{C}$ and $U$ is unitary or $K=\mathbb{R}$ and $U$ is orthogonal.

We say that a matrix embedding potential $N$ is torus invariant if $N(\lambda_1 A_1,\dots,\lambda_r A_r)=N(A_1,\dots,A_r)$ whenever
$|\lambda_1|=\dots=|\lambda_r|$.

If $A_1,\dots,A_r\in M_d(K)$, then define the completely positive superoperator $\Phi(A_1,\dots,A_r):M_d(K)\rightarrow M_d(K)$ by letting
$\Phi(A_1,\dots,A_r)(X)=A_1XA_1^\ast+\dots+A_rXA_r^\ast$. We say that a matrix embedding potential $N$ is superoperator invariant if $N(A_1,\dots,A_r)=N(B_1,\dots,B_r)$ whenever $\Phi(A_1,\dots,A_r)=\Phi(B_1,\dots,B_r)$.

We have the following examples of matrix embedding potentials.

Example 0: $N_0(A_1,\dots,A_r)=\rho(\Phi(A_1,\dots,A_r))^{1/2}$ is a superoperator invariant matrix embedding potential.

Example 1: $N_{1,p}(A_1,\dots,A_r)=\\|A_1A_1^\ast+\dots+A_rA_r^\ast\\|\_p^{1/2}$ is a matrix embedding potential for $1 < p\leq\infty$.

Example 2: $N_{2,p,q}(A_1,\dots,A_r)=\\|(A_1A_1^\ast)^q+\dots+(A_rA_r^\ast)^q\\|\_p^{1/(2q)}$ is a matrix embedding potential for $1 < p\leq\infty$ and $1\leq q < \infty.$

$L(N,a_1\dots a_n,f)=\frac{\rho(f(a_1)\dots f(a_n))^{1/n}}{N((f(a))\_{a\in A})}$.

We say that a function $f:A\rightarrow M_d(K)$ is a MPO word pre-embedding for the matrix embedding potential $N$ and corpus $a_1\dots a_n$ if the quantity $L(N,a_1\dots a_n,f)$ is a local maximum. In a MPO word pre-embedding, the matrix embedding potential $N$ both prevents $L(N,a_1\dots a_n,f)$ from being optimized simply by setting each $f(a)$ to a large value and from all of the matrices of the form $f(a)$ from being too close together. 

We observe that if $N$ is a torus invariant and $f,g:A\rightarrow M_d(K)$ functions where for all $a\in A$, there is some $\lambda_a\in S^1$ with
$f(a)=\lambda_a g(a)$, then $f$ is an MPO word pre-embedding for $N$ and $a_1\dots a_n$ if and only if $g$ is an MPO word pre-embedding for $N$ and $a_1\dots a_n$. Therefore, given a pre-word embedding $g$, we would like to select good constants $\lambda_a$ so that $f$ will be an optimal MPO word pre-embedding. We would like for the choice of system of constants $(\lambda_{a})\_{a\in A}$ to be unique and for the mapping $g\mapsto(\lambda_{a})\_{a\in A}$ to have few singularities.  For example, if $\lambda\_{a}=\frac{|\text{Tr}(g(a))|}{\text{Tr}(g(a))}$, then this choice of constants $\lambda_a$ will have a singularity of small codimension for each $a\in A$, so such a choice of $\lambda_{a}$ is unsatisfactory.

**MPOs vs LSRDRs**



**A measure of local correctness**

Suppose that $\frac{1}{r}+\frac{1}{s}=\frac{1}{t}$. Then recall that
$\|RS\|\_t\leq\|R\|\_r\cdot\|S\|\_s$. More generally, if $\frac{1}{r_1}+\dots+\frac{1}{r_h}=\frac{1}{s}$, then
$\|R_1\dots R_h\|\_s\leq\|R_1\|\_{r_1}\cdots\|R_h\|\_{r_h}$. 

If $f:A\rightarrow M_d(K)$ is an MPO word embedding, then define $Ag((r_1,\dots,r_h),f,b_1\dots b_h)$
$=\frac{\|f(b_1)\dots f(b_h)\|\_r}{\|f(b_1)\|\_{r_1}\dots\|f(b\_h)\|\_{r_h}})^{1/(h-1)}$ where $\frac{1}{r_1}+\dots+\frac{1}{r_h}=\frac{1}{r}$.

The coefficient $\mathbf{Ag}((r_1,\dots,r_h),f,b_1\dots b_h)$ is a measure of how similar the string $b_1\dots b_h$ is to strings that appear in the corpus. The coefficient $\mathbf{Ag}((r_1,\dots,r_h),f,b_1\dots b_h)$ may even be used to compare strings of different lengths.

Define $\mathbf{Ag}(f,b_1\dots b_h)=\mathbf{Ag}((\infty,\dots,\infty),f,b_1\dots b_h)$.

Recall that Von Neumann's trace inequality states that if $R,S$ have singular values $\alpha_1\geq\dots\geq\alpha_d$ and $\beta_1\geq\dots\geq\beta_d$, then
$|\text{Tr}(RS)|\leq\sum_{k=1}^d\alpha_k\beta_k$.

**Contextual embedding**

Suppose that $f:A\rightarrow M_d(K)$ is a MPO word embedding. Let $b_1\dots b_h\in A^\ast$. Let $u,v$ be column vectors. Then whenever
$1\leq j\leq h$, the left context vector with left bound $u$ of the $j$-th position in $b_1\dots b_h$ is $\frac{u^\ast f(b_1)\dots f(b_j)}{\|u^\ast f(b_1)\dots f(b_j)}\|$ and the right context vector with right bound $v$ of the $j$-th position in $b_1\dots b_h$ is $\frac{f(b_j)\dots f(b_h)v}{\|f(b_j)\dots f(b_h)v\|}$. The context pair of the $j$-th position with bounds $(u,v)$ is the pair
$(\frac{u^\ast f(b_1)\dots f(b_j)}{\|u^\ast f(b_1)\dots f(b_j)\|},\frac{f(b_j)\dots f(b_h)v}{\|f(b_j)\dots f(b_h)v\|})$.

Suppose that $f(a_j)$ has singular value decomposition $\sum_k\sigma_ku_kv_k^\ast$. Suppose now that the $j-1$-th left context vector is $\alpha^\ast$ and the $j+1$-th right context vector is $\beta$.


**Low rank matrices**

In practice, if $f:A\rightarrow M_d(K)$ is an MPO word embedding, then computer experiments indicate that the matrices $f(a)$ will be near matrices of low rank. One can therefore reduce the computation required to train, store, and use a word embedding. In particular, if $f:A\rightarrow M_d(K)$ is an MPO word embedding (or an MPO word embedding in training), then one factor each $f(a)$ as $f(a)=g(a)h(a)$ where $g(a)\in M_{d,d_a}(K),h(a)\in M_{d_a,d}(K)$ for some $d_a\leq d$.


**Graph and Markov chain embeddings**

MPO word embeddings can be used to derive graph embeddings in multiple ways.

Suppose that $(\mathcal{X}\_n)\_{n=0}^{\infty}$ is a Markov chain with a finite state space $X$. Suppose that $G:X\rightarrow A$ is a surjective function. Then for all $N>0$, the sequence $G(\mathcal{X}\_0)\dots G(\mathcal{X}\_N)$ is a word in $A^\ast$ which induces a word embedding $f:A\rightarrow M_d(K)$. Since random walks on graphs form a Markov chain, we can embed the vertices of a graph into $M_d(K)$ using our MPO word embedding obtained from the random walk.

Let $(V,E)$ be a undirected graph with possible loops. Let $0<\epsilon<1$. Now consider the Markov chain $(\mathcal{X}\_n)\_{n=0}^{\infty}$ where
$\mathcal{X}\_n=(\\{a,b\\},a)$ for some $\\{a,b\\}\in E$ with probability $1$. Then one transitions from $(\\{a,b\\},a)$ to $(\\{a,b\\},b)$ with probability $1-\epsilon$ and one transitions from $(\\{a,b\\},a)$ to randomly selected pair $(\\{c,d\\},c)$ with probability $\epsilon$. Informally, by setting $\epsilon$ to an infinitesimal, the induced embedding $f:V\rightarrow M_d(K)$ is simply a function that locally maximizes the quantity $\frac{\prod_{\\{u,v\\}\in E}\rho(f(u)f(v))^{1/(2|E|)}}{N((f(v))\_{v\in V})}$ and where the function $S_1^V\rightarrow\mathbb{R},(\lambda_v)\_{v\in V}\mapsto \rho(\sum_{k\in K}\lambda_vf(v))$ attains a local maximum when $\lambda_v=1$ for $v\in V$, and such an induced embedding shall be called a limit induced MPO embedding.


For bipartite graphs, we generalize the limit induced MPO embedding to the case when the matrices associated with each vertex are rectangular and non-square matrices. In this generalization, the non-square matrices cannot be approximated by matrices of lower rank. In other words, these rectangular matrices do not need to be compressed by factoring any further in order to save space.

If $(V,E)$ is a bipartite graph with bipartition $A,B$, then let $e\leq d$. Let $f:A\rightarrow M_{d,e}(K),g:A\rightarrow M_{e,d}(K)$ be functions. Then we say that $(f,g)$ is a RBLIMPO (rectangular bipartite limit induced matrix product optimized) pre-embedding if the quantity
$\frac{(\prod_{a\in A,b\in B,\{a,b\}\in E}\rho(f(a)g(b)))^{1/|E|}}{N((f(a))\_{a\in A},(g(b))\_{b\in B})}$ is locally maximized.

We say that a RBLIMPO pre-embedding is in-phase if $\lambda_a=1,\mu_b=1$ for $a\in A,b\in B$ is a local maximum on the set $S_1^A\times S_1^B$ of the function
$\rho(\sum_{a\in A}\lambda_a f(a)\sum_{b\in B}\mu_bg(b))$, and an in-phase RBLIMPO pre-embedding shall be called an RBLIMPO embedding.


**Limitations**


**Questions and further research**

