#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static double g_last_best_sl = 0.0;

// Min-heap for (degree, node)
typedef struct {
    int degree;
    int node;
} HeapItem;

typedef struct {
    HeapItem *data;
    int size;
    int cap;
} MinHeap;

static int heap_less(const HeapItem *a, const HeapItem *b) {
    if (a->degree != b->degree) return a->degree < b->degree;
    return a->node < b->node;
}

static void heap_init(MinHeap *h, int cap) {
    h->size = 0;
    h->cap = cap > 0 ? cap : 16;
    h->data = (HeapItem *)malloc((size_t)h->cap * sizeof(HeapItem));
}

static void heap_free(MinHeap *h) {
    free(h->data);
    h->data = NULL;
    h->size = 0;
    h->cap = 0;
}

static void heap_grow(MinHeap *h) {
    int new_cap = h->cap * 2;
    HeapItem *new_data = (HeapItem *)realloc(h->data, (size_t)new_cap * sizeof(HeapItem));
    if (!new_data) return; // OOM: keep old, caller will likely fail later
    h->data = new_data;
    h->cap = new_cap;
}

static void heap_push(MinHeap *h, HeapItem item) {
    if (h->size >= h->cap) heap_grow(h);
    int i = h->size++;
    h->data[i] = item;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!heap_less(&h->data[i], &h->data[p])) break;
        HeapItem tmp = h->data[i];
        h->data[i] = h->data[p];
        h->data[p] = tmp;
        i = p;
    }
}

static HeapItem heap_pop(MinHeap *h) {
    HeapItem out = h->data[0];
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            int r = 2 * i + 2;
            int smallest = i;
            if (l < h->size && heap_less(&h->data[l], &h->data[smallest])) smallest = l;
            if (r < h->size && heap_less(&h->data[r], &h->data[smallest])) smallest = r;
            if (smallest == i) break;
            HeapItem tmp = h->data[i];
            h->data[i] = h->data[smallest];
            h->data[smallest] = tmp;
            i = smallest;
        }
    }
    return out;
}

// DSU with component size and Q
typedef struct {
    int *parent;
    int *size;
    unsigned char *made;
    double *Q;
} DSU;

static void dsu_init(DSU *d, int n) {
    d->parent = (int *)calloc((size_t)(n + 1), sizeof(int));
    d->size = (int *)calloc((size_t)(n + 1), sizeof(int));
    d->made = (unsigned char *)calloc((size_t)(n + 1), sizeof(unsigned char));
    d->Q = (double *)calloc((size_t)(n + 1), sizeof(double));
}

static void dsu_free(DSU *d) {
    free(d->parent);
    free(d->size);
    free(d->made);
    free(d->Q);
    d->parent = NULL;
    d->size = NULL;
    d->made = NULL;
    d->Q = NULL;
}

static void dsu_make(DSU *d, int v) {
    if (!d->made[v]) {
        d->made[v] = 1;
        d->parent[v] = v;
        d->size[v] = 1;
        d->Q[v] = 0.0;
    }
}

static int dsu_find(DSU *d, int v) {
    if (!d->made[v]) return v;
    int root = v;
    while (d->parent[root] != root) root = d->parent[root];
    while (d->parent[v] != v) {
        int p = d->parent[v];
        d->parent[v] = root;
        v = p;
    }
    return root;
}

static int dsu_union(DSU *d, int a, int b) {
    dsu_make(d, a);
    dsu_make(d, b);
    int ra = dsu_find(d, a);
    int rb = dsu_find(d, b);
    if (ra == rb) return ra;
    if (d->size[ra] < d->size[rb]) {
        int t = ra; ra = rb; rb = t;
    }
    d->parent[rb] = ra;
    d->size[ra] += d->size[rb];
    d->Q[ra] += d->Q[rb];
    return ra;
}

static int *g_idx = NULL;
static int cmp_by_idx(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    int da = g_idx[ia];
    int db = g_idx[ib];
    return (da < db) ? -1 : (da > db);
}

static long long sum_succ_until(int v, int T, const int *succ_offsets, const int *succ_adj, const int *idx, const int *deg) {
    long long s = 0;
    for (int pos = succ_offsets[v]; pos < succ_offsets[v + 1]; pos++) {
        int w = succ_adj[pos];
        if (idx[w] >= T) break;
        s += (long long)deg[w];
    }
    return s;
}

// Core API: adjacency in CSR with offsets[1..n+1], neighbors[0..offsets[n+1]-1]
// Returns count of nodes in best component; writes them into out_nodes (capacity out_cap >= n)
int lacore_run(int n, const int *offsets, const int *neighbors, double eps, int *out_nodes, int out_cap) {
    if (!offsets || !neighbors || !out_nodes || n <= 0) return -1;
    if (out_cap < n) return -2;

    // Phase 1: peeling
    int *deg0 = (int *)calloc((size_t)(n + 1), sizeof(int));
    if (!deg0) return -3;

    int total_adj = offsets[n + 1];
    (void)total_adj; // unused, but indicates adjacency size

    int heap_cap = n * 4 + 16;
    MinHeap heap;
    heap_init(&heap, heap_cap);

    for (int i = 1; i <= n; i++) {
        deg0[i] = offsets[i + 1] - offsets[i];
        heap_push(&heap, (HeapItem){deg0[i], i});
    }

    int *peel_stack = (int *)malloc((size_t)n * sizeof(int));
    int peel_top = 0;

    while (heap.size > 0) {
        HeapItem p = heap_pop(&heap);
        if (p.degree != deg0[p.node]) continue; // stale
        peel_stack[peel_top++] = p.node;
        for (int pos = offsets[p.node]; pos < offsets[p.node + 1]; pos++) {
            int v = neighbors[pos];
            if (deg0[v] > 0) {
                deg0[v] -= 1;
                heap_push(&heap, (HeapItem){deg0[v], v});
            }
        }
        deg0[p.node] = 0;
    }

    heap_free(&heap);

    int *add_order = (int *)malloc((size_t)n * sizeof(int));
    int *idx = (int *)malloc((size_t)(n + 1) * sizeof(int));
    for (int t = 0; t < n; t++) {
        int u = peel_stack[--peel_top];
        add_order[t] = u;
        idx[u] = t;
    }
    free(peel_stack);

    // Phase 1.5: orient edges
    int *succ_counts = (int *)calloc((size_t)(n + 1), sizeof(int));
    int *pred_counts = (int *)calloc((size_t)(n + 1), sizeof(int));

    for (int u = 1; u <= n; u++) {
        for (int pos = offsets[u]; pos < offsets[u + 1]; pos++) {
            int v = neighbors[pos];
            if (u < v) {
                if (idx[u] < idx[v]) {
                    succ_counts[u] += 1;
                    pred_counts[v] += 1;
                } else {
                    succ_counts[v] += 1;
                    pred_counts[u] += 1;
                }
            }
        }
    }

    int *succ_offsets = (int *)malloc((size_t)(n + 2) * sizeof(int));
    int *pred_offsets = (int *)malloc((size_t)(n + 2) * sizeof(int));
    int succ_total = 0;
    int pred_total = 0;
    for (int i = 1; i <= n; i++) {
        succ_offsets[i] = succ_total;
        pred_offsets[i] = pred_total;
        succ_total += succ_counts[i];
        pred_total += pred_counts[i];
    }
    succ_offsets[n + 1] = succ_total;
    pred_offsets[n + 1] = pred_total;

    int *succ_adj = (int *)malloc((size_t)succ_total * sizeof(int));
    int *pred_adj = (int *)malloc((size_t)pred_total * sizeof(int));
    int *succ_pos = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *pred_pos = (int *)malloc((size_t)(n + 1) * sizeof(int));
    for (int i = 1; i <= n; i++) {
        succ_pos[i] = succ_offsets[i];
        pred_pos[i] = pred_offsets[i];
    }

    for (int u = 1; u <= n; u++) {
        for (int pos = offsets[u]; pos < offsets[u + 1]; pos++) {
            int v = neighbors[pos];
            if (u < v) {
                if (idx[u] < idx[v]) {
                    succ_adj[succ_pos[u]++] = v;
                    pred_adj[pred_pos[v]++] = u;
                } else {
                    succ_adj[succ_pos[v]++] = u;
                    pred_adj[pred_pos[u]++] = v;
                }
            }
        }
    }

    free(succ_pos);
    free(pred_pos);
    free(succ_counts);
    free(pred_counts);

    // sort successors by idx
    g_idx = idx;
    for (int v = 1; v <= n; v++) {
        int len = succ_offsets[v + 1] - succ_offsets[v];
        if (len > 1) {
            qsort(succ_adj + succ_offsets[v], (size_t)len, sizeof(int), cmp_by_idx);
        }
    }

    // Phase 2: reverse reconstruction
    DSU dsu;
    dsu_init(&dsu, n);
    int *deg = (int *)calloc((size_t)(n + 1), sizeof(int));
    long long *pred_sum = (long long *)calloc((size_t)(n + 1), sizeof(long long));

    double bestSL = 0.0;
    int best_count = 0;

    for (int t = 0; t < n; t++) {
        int u = add_order[t];
        dsu_make(&dsu, u);
        int ru = dsu_find(&dsu, u);
        double sL = dsu.size[ru] / (dsu.Q[ru] + eps);
        if (sL > bestSL) {
            bestSL = sL;
            best_count = 0;
            for (int i = 1; i <= n; i++) {
                if (dsu.made[i] && dsu_find(&dsu, i) == ru) {
                    out_nodes[best_count++] = i;
                }
            }
        }

        long long Su = 0;
        int Tu = idx[u];

        for (int pos = pred_offsets[u]; pos < pred_offsets[u + 1]; pos++) {
            int v = pred_adj[pos];
            long long a = deg[u];
            long long b = deg[v];
            long long Sv = pred_sum[v] + sum_succ_until(v, Tu, succ_offsets, succ_adj, idx, deg);

            long long dQu = 2 * a * a - 2 * Su + a;
            long long dQv = 2 * b * b - 2 * Sv + b;
            long long diff = a - b;
            long long edgeTerm = diff * diff;

            int r_u = dsu_find(&dsu, u);
            int r_v = dsu_find(&dsu, v);

            dsu.Q[r_u] += (double)dQu;
            dsu.Q[r_v] += (double)dQv;

            int r;
            if (r_u != r_v) {
                r = dsu_union(&dsu, r_u, r_v);
                dsu.Q[r] += (double)edgeTerm;
            } else {
                r = r_u;
                dsu.Q[r] += (double)edgeTerm;
            }

            sL = dsu.size[r] / (dsu.Q[r] + eps);
            if (sL > bestSL) {
                bestSL = sL;
                best_count = 0;
                for (int i = 1; i <= n; i++) {
                    if (dsu.made[i] && dsu_find(&dsu, i) == r) {
                        out_nodes[best_count++] = i;
                    }
                }
            }

            deg[u] += 1;
            deg[v] += 1;

            for (int p2 = succ_offsets[u]; p2 < succ_offsets[u + 1]; p2++) {
                int y = succ_adj[p2];
                pred_sum[y] += 1;
            }
            for (int p2 = succ_offsets[v]; p2 < succ_offsets[v + 1]; p2++) {
                int y = succ_adj[p2];
                pred_sum[y] += 1;
            }

            Su += deg[v];
        }
    }

    free(add_order);
    free(idx);
    free(succ_offsets);
    free(pred_offsets);
    free(succ_adj);
    free(pred_adj);
    free(deg0);
    free(deg);
    free(pred_sum);
    dsu_free(&dsu);

    g_last_best_sl = bestSL;
    return best_count;
}

double lacore_last_best_sl(void) {
    return g_last_best_sl;
}
