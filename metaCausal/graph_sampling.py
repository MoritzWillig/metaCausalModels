import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def is_acyclic(adj: np.ndarray) -> bool:
    # unsure if row- or column-major is expected, but a cyclic graph is cyclic in both cases :)
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(G)


def select_n_edges(adj, n, rng):
    nadj = np.zeros_like(adj)
    if n == 0:
        return nadj

    existing_entries_a, existing_entries_b = np.nonzero(adj)
    selected_edge_idcs = rng.choice(len(existing_entries_a), n, replace=False)
    selected_entries_a = existing_entries_a[selected_edge_idcs]
    selected_entries_b = existing_entries_b[selected_edge_idcs]

    nadj[selected_entries_a, selected_entries_b] = 1
    return nadj


def alter_adj_mat(adj, num_new_edges, num_altered_edges, num_removed_edges, rng):
    adj = np.copy(adj)

    rem_adj = select_n_edges(adj, num_removed_edges, rng)
    alt_adj = select_n_edges(adj-rem_adj, num_altered_edges, rng)  # alter edges that exist and do not get removed

    # add some edges that do not exist yet
    # adding edges is also allowed in the upper triangular area (as it might imply a change in row order), we need
    # to allow for this and check admissibility...

    add_adj = np.zeros_like(adj)
    if num_new_edges != 0:
        # all points except diagonal, existing edges and already added edges (removed edges might not be readded)
        candidates = 1 - np.eye(adj.shape[0], adj.shape[1]) - adj - add_adj
        candidates_a, candidates_b = np.nonzero(candidates)
        perm = rng.permutation(len(candidates_a))  # pick a random order of candidates to try to add

        base_adj = adj - rem_adj
        num_added_candidates = 0
        for j in range(len(candidates_a)):
            selected_entry_a = candidates_a[perm[j]]
            selected_entry_b = candidates_b[perm[j]]
            add_adj[selected_entry_a, selected_entry_b] = 1

            if not is_acyclic(base_adj + add_adj):
                # edge addition leads to a cycle -> revert modification
                add_adj[selected_entry_a, selected_entry_b] = 0
            else:
                num_added_candidates += 1
                if num_added_candidates == num_new_edges:
                    break
        else:
            print(f"Unable to add the desired number of edges ({num_added_candidates}/{num_new_edges}).")

    return add_adj, alt_adj, rem_adj


def sample_meta_graph(
        num_vars, num_edges, num_meta_states,
        num_max_new_edges, num_max_altered_edges, num_max_removed_edges,
        min_edge_weight, max_edge_weight,
        rng: np.random.Generator):
    # A_{i,j} -> j is a parent of i
    # create lower triangular matrix (zeroed out main diagonal) and reduce to n-percent of max edges
    full_adj_mat = np.tri(num_vars, num_vars, -1)
    desired_num_edges = num_edges  # np.round(max_num_edges * edge_percentage)
    base_adj_mat = select_n_edges(full_adj_mat, desired_num_edges, rng)

    # create meta states
    meta_adj = np.zeros([num_vars, num_vars, num_meta_states])
    meta_adj_detail = np.zeros([num_vars, num_vars, num_meta_states, 3])  # [N,N,M,(new,alt,rem)]

    # initialize base meta state
    meta_adj[:,:,0] = base_adj_mat
    meta_adj_detail[:, :, 0, 1] = base_adj_mat  # makes sure that we later on sample weights for base matrix

    for i in range(1, num_meta_states):
        add_adj, alt_adj, rem_adj = alter_adj_mat(
            base_adj_mat,
            rng.integers(0, num_max_new_edges+1),
            rng.integers(0, num_max_altered_edges+1),
            rng.integers(0, num_max_removed_edges+1),
            rng)

        meta_state_adj_mat = base_adj_mat + add_adj - rem_adj
        meta_adj[:, :, i] = meta_state_adj_mat
        meta_adj_detail[:, :, i, 0] = add_adj
        meta_adj_detail[:, :, i, 1] = alt_adj
        meta_adj_detail[:, :, i, 2] = rem_adj

    # sample initial weights
    def sample_weights():
        return (rng.uniform(min_edge_weight, max_edge_weight, size=base_adj_mat.shape) *
                    rng.choice([-1, 1], replace=True, size=base_adj_mat.shape))

    base_weights = sample_weights()
    meta_adj *= base_weights[:, :, None]

    # set altered mechanism weights
    for i in range(1, num_meta_states):
        meta_adj[:, :, i] = (1 - meta_adj_detail[:, :, i, 1]) * meta_adj[:, :, i] + (meta_adj_detail[:, :, i, 1] * sample_weights())

    return meta_adj, meta_adj_detail
