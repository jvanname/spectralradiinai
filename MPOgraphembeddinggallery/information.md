Here, I post images obtained from limit induced MPO graph embeddings. We observe that limit induced MPO graph embeddings are not designed for image processing; unlike CNNs, limit induced MPO graph embeddings do not take into account the ordering or the distance between rows or columns of pixels. These images however visually illustrate the ability for limit MPO graph embeddings to learn which pairs of nodes are likely to be an edge and which pairs of nodes are not likely to form an edge.

Discontinuity: Since other social media platforms are better suited for posting images, I will try to refrain from posting any more media here.

Suppose that $G=(\{1,\dots,n\},E)$ is an undirected graph. Then we can represent the limit induced MPO graph pre-embedding $f$ of $V$ by the heatmap of the function $g\:\\{1,\dots,n\\}^2\rightarrow \[0,\infty)$ where $g(a,b)=\rho(f(a)f(b))$ for $a,b\in\{1,\dots,n\}$. Suppose that $(\{1,\dots,m\},\{1,\dots,n\},E)$ is an undirected bipartite graph. Then we can represent the limit induced MPO graph pre-embedding $f$ of $V$ by the heatmap of the function $h:\\{1,\dots,m\\}\times\\{1,\dots,n\\}\rightarrow \[0,\infty)$ where $h(a,b)=\rho(f(a)f(b))$ for $a,b\in\{1,\dots,n\}$.

Observations:

1. The function $g$ constructed above resembles the nearest low-rank matrix (in the Frobenius norm) to the adjacency matrix of the graph $G$ (bi-adjacency matrix for bipartite graphs). For example, the graph $(\\{1,\dots,n\\},E)$ where $E=\\{(x,y)\in\\{1,\dots,n\\}^2\mid \sin((x^2+y^2)/r)>0\\}$ is easy for a limit induced MPO graph embedding to recognize since the gradient ascent algorithm learns to constructively interfere on the set $E$ and as a consequence, the gradient ascent algorithm also destructively interferes on the set $\\{1,\dots,n\\}^2\setminus E,$ but the adjacency matrix of $E$ is approximately $(\sin((i^2+j^2)/2))\_{i,j}$ which has rank 2. Since the adjacency matrix of $E$ approximately a low rank-matrix, the closest rank 2 matrix in the Frobenius norm to the adjacency matrix of $E$ is a good approximation.

2. The functions $f,h$ can be made to fit (and overfit) any undirected or bipartite graph.

3. Compared to other LSRDRs, limit induced MPO graph embeddings do not converge to equivalent local maxima when when one repeats the gradient ascent with different initial conditions. In other words, the images produced here depend on the initial conditions chosen. Fortunately, we have not found any initial conditions in which the corresponding gradient ascent limit induced MPO graph embeddings converges to a bad output.
