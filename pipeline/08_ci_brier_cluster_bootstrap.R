# =============================================================================
# 08_ci_brier_cluster_bootstrap.R
#   Confidence intervals for the chal_health Brier of the candidate score sets.
# =============================================================================

# -----------------------------------------------------------------------------
# Cluster bootstrap. `pred_list` is a named list of per-row predictions already
# in the desired space (raw or nested-calibrated), each aligned to `y`/`doi_o`.
# -----------------------------------------------------------------------------
compute_brier_cis <- function(pred_list, y, doi_o, in_target,
                              B = 2000L, conf = 0.95, ref = NULL, seed = 1L) {
  keys  <- names(pred_list)
  a     <- (1 - conf) / 2
  qlohi <- function(v) stats::quantile(v, c(a, 1 - a), na.rm = TRUE, names = FALSE)

  base_ok  <- in_target & is.finite(y)          # candidate finiteness handled per draw
  clusters <- unique(doi_o[base_ok])
  rows_by_cluster <- split(which(base_ok), doi_o[base_ok])

  brier_at <- function(p, rows) {               # p, y indexed by row; rows may repeat
    yy <- y[rows]; pp <- p[rows]
    ok <- is.finite(pp) & is.finite(yy)
    if (sum(ok) < 3L) NA_real_ else mean((pp[ok] - yy[ok])^2)
  }

  point <- vapply(keys, function(k) brier_at(pred_list[[k]], which(base_ok)), numeric(1))

  pairs <- if (is.null(ref)) utils::combn(keys, 2L, simplify = FALSE)
           else lapply(setdiff(keys, ref), function(k) c(ref, k))

  boot_m <- matrix(NA_real_, B, length(keys),  dimnames = list(NULL, keys))
  boot_d <- matrix(NA_real_, B, length(pairs))

  set.seed(seed)
  for (b in seq_len(B)) {
    samp <- sample(clusters, length(clusters), replace = TRUE)
    rows <- unlist(rows_by_cluster[as.character(samp)], use.names = FALSE)
    yy   <- y[rows]
    for (j in seq_along(keys))
      boot_m[b, j] <- brier_at(pred_list[[keys[j]]], rows)
    for (m in seq_along(pairs)) {               # paired: SAME rows, both finite
      pa <- pred_list[[pairs[[m]][1]]][rows]; pb <- pred_list[[pairs[[m]][2]]][rows]
      ok <- is.finite(pa) & is.finite(pb) & is.finite(yy)
      boot_d[b, m] <- if (sum(ok) >= 3L)
        mean((pa[ok] - yy[ok])^2) - mean((pb[ok] - yy[ok])^2) else NA_real_
    }
  }

  marginal <- tibble::tibble(
    candidate = keys, brier = as.numeric(point),
    ci_lo = apply(boot_m, 2, function(v) qlohi(v)[1]),
    ci_hi = apply(boot_m, 2, function(v) qlohi(v)[2]))

  paired <- tibble::tibble(
    a = vapply(pairs, `[`, character(1), 1),
    b = vapply(pairs, `[`, character(1), 2),
    diff_brier = vapply(seq_along(pairs), function(m)
      point[pairs[[m]][1]] - point[pairs[[m]][2]], numeric(1)),
    ci_lo = apply(boot_d, 2, function(v) qlohi(v)[1]),
    ci_hi = apply(boot_d, 2, function(v) qlohi(v)[2])) |>
    dplyr::mutate(excludes_0 = (ci_lo > 0) | (ci_hi < 0))   # TRUE = distinguishable at conf

  list(n_rows = sum(base_ok), n_clusters = length(clusters),
       marginal = marginal, paired = paired)
}

