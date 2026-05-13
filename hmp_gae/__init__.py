# hmp_gae package
# Hypergraph Message-Passing Graph AutoEncoder for FedLLM immunization.
#
# Sub-modules:
#   - node_features : eta_i = f_enc(Delta_i, stats, history)
#   - hypergraph    : k-NN hypergraph construction H, D_V, D_E
#   - encoder       : L-layer HMP encoder (node <-> hyperedge <-> node)
#   - decoder       : inner-product adjacency decoder + hyperedge decoder
#   - losses        : BCE(H, H_hat) + smoothness(Z, A_hat) + hist(Z, Z_hist)
#   - trust_scorer  : closed-form s_i -> alpha_i = softmax(s_i / tau)
#   - runtime       : HMPGAERuntime wiring everything together (used by defense package)
