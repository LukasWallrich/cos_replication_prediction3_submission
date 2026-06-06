# =============================================================================
# 07_calibration.R  —  Probability calibration toolkit
# =============================================================================
#
# WHY THIS MATTERS HERE
# ---------------------
# The Round 3 test set is n = 45 and the metric is Brier. With so few rows, a
# poorly-calibrated probability is punished hard: an over-confident 0.9 that
# turns out to be a 0 costs 0.81. Tree ensembles (RF, XGB) in particular
# produce probabilities that are systematically too extreme. A calibration map
# fit on held-out data pulls predictions back toward their empirical frequency
# and is usually the single highest-leverage post-processing step at small n.
#
# CONTRACT
# --------
#   fit_calibrator(probs, labels, method)  -> calibrator object (list w/ $method)
#   apply_calibrator(calibrator, probs)    -> calibrated probs in [0,1]
#
# All calibrators MUST be fit on data the base/stacked model did NOT train on
# (an out-of-fold or held-out split); otherwise calibration overfits and looks
# better in-sample than it is. compare_calibrators() enforces this by taking
# separate fit/eval inputs.
#
# Methods:
#   "none"        identity (baseline)
#   "temperature" single scalar T on the logit scale (monotone, 1 param)
#   "platt"       logistic regression of label ~ logit(prob) (2 params)
#   "isotonic"    non-parametric monotone step map (flexible, needs more data)
#
# Robustness: every fitter degrades gracefully to "none" when there are too few
# finite pairs or only one outcome class — it never errors out a pipeline run.
# =============================================================================

suppressPackageStartupMessages({
  source(here::here("pipeline/00_packages_config.R"))
})

# Minimum finite (prob, label) pairs required to attempt calibration.
.CALIB_MIN_N <- 8L

# Clip helper: keep probabilities strictly inside (0,1) for logit stability.
.clip01 <- function(p, eps = 1e-10) pmax(pmin(as.numeric(p), 1 - eps), eps)

# ── Temperature scaling ────────────────────────────────────────────────────────
# Finds scalar T minimising NLL on a held-out set; T > 1 softens, T < 1 sharpens.
temperature_scaling <- function(probs, labels) {
  probs  <- .clip01(probs)
  labels <- as.numeric(labels)
  ok     <- is.finite(probs) & is.finite(labels)
  if (sum(ok) < 3 || length(unique(round(labels[ok]))) < 2) {
    return(list(T = 1))
  }
  logits    <- qlogis(probs[ok])
  labels_ok <- labels[ok]
  nll <- function(T) {
    p <- .clip01(plogis(logits / T), 1e-15)
    -mean(labels_ok * log(p) + (1 - labels_ok) * log(1 - p))
  }
  opt <- optim(par = 1, fn = nll, method = "L-BFGS-B", lower = 0.01, upper = 20)
  list(T = opt$par)
}

# ── fit_calibrator ─────────────────────────────────────────────────────────────

#' @param probs   numeric vector of model probabilities (held-out / OOF).
#' @param labels  numeric 0/1 (or soft) outcomes aligned to `probs`.
#' @param method  one of "none", "temperature", "platt", "isotonic".
#' @return list with $method and any fitted parameters. Falls back to
#'   list(method = "none") when data is insufficient.
fit_calibrator <- function(probs, labels,
                           method = c("temperature", "none", "platt", "isotonic")) {
  method <- match.arg(method)
  probs  <- as.numeric(probs)
  labels <- as.numeric(labels)
  ok     <- is.finite(probs) & is.finite(labels)

  # Not enough signal to calibrate → identity. 
  if (method == "none" ||
      sum(ok) < .CALIB_MIN_N ||
      length(unique(round(labels[ok]))) < 2) {
    if (method != "none" && sum(ok) < .CALIB_MIN_N) {
      message(sprintf("  [calibration] only %d usable pairs (<%d) — using identity.",
                      sum(ok), .CALIB_MIN_N))
    }
    return(list(method = "none"))
  }

  p <- .clip01(probs[ok])
  y <- labels[ok]

  switch(method,
    "temperature" = {
      ts <- temperature_scaling(p, y)
      list(method = "temperature", T = ts$T)
    },
    "platt" = {
      d   <- data.frame(.logit = qlogis(p), .y = y)
      fit <- suppressWarnings(
        glm(.y ~ .logit, data = d, family = quasibinomial())
      )
      list(method = "platt", coef = stats::coef(fit))
    },
    "isotonic" = {
      o  <- order(p)
      ps <- p[o]; ys <- y[o]
      ir <- stats::isoreg(ps, ys)
      xs <- ir$x; yf <- ir$yf
      # approxfun needs unique x; keep the last (largest) fitted value per tie
      # so the step map stays monotone non-decreasing.
      keep <- !duplicated(xs, fromLast = TRUE)
      fn <- if (sum(keep) >= 2) {
        stats::approxfun(xs[keep], yf[keep], method = "linear", rule = 2)
      } else {
        local({ const <- mean(yf); function(x) rep(const, length(x)) })
      }
      list(method = "isotonic", fn = fn)
    }
  )
}

# ── apply_calibrator ─────────────────────────────────────────────────────────

apply_calibrator <- function(calibrator, probs) {
  probs <- as.numeric(probs)
  out <- switch(calibrator$method,
    "none"        = probs,
    "temperature" = plogis(qlogis(.clip01(probs)) / calibrator$T),
    "platt"       = {
      b <- calibrator$coef
      plogis(b[[1]] + b[[2]] * qlogis(.clip01(probs)))
    },
    "isotonic"    = calibrator$fn(.clip01(probs)),
    stop("unknown calibrator method: ", calibrator$method)
  )
  pmax(pmin(as.numeric(out), 1), 0)
}

# ── compare_calibrators ────────────────────────────────────────────────────────
# Validation helper: fit each method on (fit_probs, fit_labels) and score Brier
# on a DISJOINT (eval_probs, eval_labels). This is the only honest way to judge
# calibration — fitting and scoring on the same rows always flatters isotonic.
compare_calibrators <- function(fit_probs, fit_labels, eval_probs, eval_labels,
                                methods = c("none", "temperature", "platt", "isotonic")) {
  purrr::map_dfr(methods, function(m) {
    cal  <- fit_calibrator(fit_probs, fit_labels, method = m)
    ep   <- apply_calibrator(cal, eval_probs)
    tibble::tibble(
      method        = m,
      method_used   = cal$method,            # "none" if a method degraded
      brier_eval    = brier_score(ep, eval_labels),
      mean_pred     = mean(ep, na.rm = TRUE)
    )
  })
}

cat("[07_calibration.R] loaded: fit_calibrator, apply_calibrator,",
    "compare_calibrators (none/temperature/platt/isotonic)\n")

# ── Standalone self-test ───────────────────────────────────────────────────────
# Rscript pipeline/07_calibration.R  → sanity check on synthetic over-confident
# probabilities. Only runs when this file is the script passed to Rscript, not
# when it is source()'d by another pipeline step.
.calib_is_main <- {
  fa <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  length(fa) == 1 && basename(fa) == "07_calibration.R"
}
if (.calib_is_main) {
  set.seed(CONFIG$seed)
  n <- 2000
  true_p   <- plogis(rnorm(n))
  y        <- rbinom(n, 1, true_p)
  reported <- plogis(qlogis(true_p) * 1.8)   # over-confident: pushed toward 0/1
  idx <- sample(n, n / 2)
  cmp <- compare_calibrators(reported[idx], y[idx], reported[-idx], y[-idx])
  cat("\nSelf-test — Brier on held-out half (lower is better):\n")
  print(cmp)
  cat("\nExpected: temperature/platt/isotonic all beat 'none'",
      "(reported probs are over-confident).\n")
}
