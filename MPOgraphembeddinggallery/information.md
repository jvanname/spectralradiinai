Here, we post images obtained from limit induced MPO graph embeddings.

We observe that limit induced MPO graph embeddings are not designed for image processing; unlike CNNs, limit induced MPO graph embeddings do not take into account the ordering or the distance between rows or columns of pixels. Instead, these images show that

We also observe that some images are easy for limit induced MPO graph embeddings to learn; for example, the graph $(\\{1,\dots,n\\},E)$ where $E=\\{(x,y)\in\\{1,\dots,n\\}^2\mid \sin((x^2+y^2)/r)>0\\}$ is easy for a limit induced MPO graph embedding to recognize since the gradient ascent algorithm learns to constructively interfere on the set $E$ and as a consequence, the gradient ascent algorithm also destructively interferes on the set $\\{1,\dots,n\\}^2\setminus E.$
