# Offline Trace Analysis & Fenwick Tree Reuse Distance

How `mp-analyze` calculates reuse distance and generates exact Miss Ratio Curves (MRC).

## O(N log N) Reuse Distance Algorithm
Calculating reuse distance naively is $O(N^2)$. `mp-analyze` maintains a Fenwick tree (Binary Indexed Tree) indexed by access position:
- Each unit access queries the tree for distinct items seen since its previous access.
- Updates insert the new access position in $O(\log N)$ time.
- Yields the exact Mattson LRU stack distance profile across millions of access events.
