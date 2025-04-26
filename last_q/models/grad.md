$$
\begin{aligned}
\tfrac12\|A\cdot B - M\|_F^2
&=\tfrac12\,\mathrm{tr}\bigl[(A\cdot B - M)^T(A\cdot B - M)\bigr]\\
&=\tfrac12\,\mathrm{tr}\bigl[(B^T\cdot A^T - M^T)(A\cdot B - M)\bigr]
\end{aligned}
$$
and knowing:
1. $\mathrm{tr}(A\cdot B\cdot C) = \mathrm{tr}(B\cdot C\cdot A) = \mathrm{tr}(C\cdot A\cdot B)$ 
2. $\mathrm{tr}(A) = \mathrm{tr}(A^T)$
3. $(A\cdot B)^T = B^T \cdot A^T$

then:
$$
\begin{aligned}
\mathrm{tr}\bigl[(B^T\cdot A^T - M^T)(A\cdot B - M)\bigr]
&=\mathrm{tr}(B^T\cdot A^T\cdot A\cdot B)\;-\;\mathrm{tr}(B^T\cdot A^T\cdot M)\;-\;\mathrm{tr}(M^T\cdot A\cdot B)\;+\;\mathrm{tr}(M^T\cdot M)\\
&=\mathrm{tr}(A\cdot B\cdot B^T\cdot A^T)\;-\;\mathrm{tr}(A\cdot B\cdot M^T)\;-\;\mathrm{tr}(A\cdot B\cdot M^T)\;+\;\mathrm{const}\\
&=\mathrm{tr}(A\cdot B\cdot B^T\cdot A^T)\;-2\;\mathrm{tr}(A\cdot B\cdot M^T)\;+\;\mathrm{const}
\end{aligned}
$$

Additionally:

1. 
  $$
  \begin{aligned}
  \mathrm{tr}(A\cdot N) &= \sum_{i} \bigl(A\cdot N\bigr)_{ii} \\
  &= \sum_{i,j} A_{ij}\cdot N_{ji}
  \end{aligned}
  $$
  $$
  \begin{aligned}
  \Rightarrow \frac{\partial}{\partial A_{pq}} \mathrm{tr}(A\cdot N) &=\frac{\partial}{\partial A_{pq}} \sum_{i,j} (A_{ij}\cdot N_{ji}) \\
  &= \frac{\partial}{\partial A_{pq}} (A_{pq}\cdot N_{qp}) \\
  &= N_{qp}
  \end{aligned}
  $$
  $$
  \Rightarrow \frac{\partial}{\partial A}\, \mathrm{tr}(A\cdot N) = N^T
  $$
2. 
  $$
  \begin{aligned}
  \mathrm{tr}(A\cdot C\cdot A^T) &= \sum_{i} \bigl(A\cdot C\cdot A^T\bigr)_{ii} \\
  &= \sum_{i,j,k} A_{ij}\cdot C_{jk} \cdot A_{ki}^T \\
  &= \sum_{i,j,k} A_{ij}\cdot C_{jk} \cdot A_{ik}
  \end{aligned}
  $$
  $$
  \begin{aligned}
  \Rightarrow \frac{\partial}{\partial A_{pq}} \mathrm{tr}(A\cdot C\cdot A^T) &= \frac{\partial}{\partial A_{pq}} \sum_{i,j,k} (A_{ij}\cdot C_{jk} \cdot A_{ik}) \\
  &= \frac{\partial}{\partial A_{pq}} (\sum_{k} (A_{pq}\cdot C_{qk} \cdot A_{pk})) + \frac{\partial}{\partial A_{pq}} (\sum_{j} (A_{pj}\cdot C_{jq} \cdot A_{pq})) \\
  &= \sum_{k} (C_{qk} \cdot A_{pk}) + \sum_{j} (A_{pj}\cdot C_{jq}) \\
  &= \sum_{k} ((C_{qk} \cdot A_{pk}) + (C_{kq} \cdot A_{kp})) \\
  &= \sum_{k} ((C + C^T)_{qk} \cdot A_{pk}) \quad C + C^T \text{ is symmetric}\\ 
  &= \sum_{k} (A_{pk} \cdot (C + C^T)_{kq}) \\
  &= [A \cdot (C + C^T)]_{pq}
  \end{aligned}
  $$
  $$
  \Rightarrow \frac{\partial}{\partial A}\, \mathrm{tr}(A\cdot C\cdot A^T) = [A \cdot (C + C^T)]
  $$

Therefore:

1. 
  $$
  \begin{aligned}
  \frac{\partial}{\partial A}\Bigl[-\tfrac12\,\mathrm{tr}(A\cdot B\cdot M^T)\Bigr] &= -\tfrac12\cdot (B\cdot M^T)^T \\
  &= -\tfrac12\cdot M\cdot B^T
  \end{aligned}
  $$
2.
  $$
  \begin{aligned}
  \frac{\partial}{\partial A}\,\tfrac12\,\mathrm{tr}(A\cdot B\cdot B^T\cdot A^T) &= \tfrac12 \cdot A\cdot (B\cdot B^T + (B\cdot B^T)^T) \\
  &= \tfrac12 \cdot A\cdot (B\cdot B^T + B\cdot B^T) \\
  &= A\cdot B\cdot B^T
  \end{aligned}
  $$

Putting it all together:
$$
\frac{\partial}{\partial A}\,\tfrac12\|A\cdot B - M\|_F^2
= A\cdot B\cdot B^T \;-\; M\cdot B^T
= (A\cdot B - M)\cdot B^T
$$

Substitute:
$$A=X\cdot h_1$$
$$B=h_2\cdot Y^T$$
and $\frac{\partial A}{\partial h1} = \frac{\partial X\cdot h_1}{\partial h1} = X^T$

Using the chain rule:
$$
  \begin{aligned}
\frac{\partial}{\partial h_1} \,\tfrac12\|X\cdot h_1\cdot h_2\cdot Y^T - M\|_F^2 &=\; \tfrac12\ \frac{\partial \, (X\cdot h_1)}{\partial h_1} \, \frac{\partial \, \|X\cdot h_1\cdot h_2\cdot Y^T - M\|_F^2}{\partial (X\cdot h_1)}  \\
&=\; X^T \cdot (X\cdot h_1 \cdot h_2\cdot Y^T - M)\cdot Y\cdot h_2^T
\end{aligned}
$$

The same analysis hold for the derivative wrt $h_2$