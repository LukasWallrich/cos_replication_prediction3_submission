# =============================================================================
# prior_shrinkage / 3_build_predictor.R
#
# Builds the deployed m2 predictor for the 45 Round 3 claims:
#   1. aggregate the R3 priors (Opus + GPT-5.5 + Gemini) -> prior_3fam
#   2. compute the evidence term p_base with the FROZEN tau2 (=0.019454) from
#      1_evidence_shrinkage.R
#   3. fit logistic(y ~ prior_3fam + p_base) on the 107 labelled health rows and
#      apply it to R3.
#
# INPUTS (challenge-derived; NOT shipped — see DATA.md):
#   data/r3_claimmap.csv, data/r3_priors/, data/r3_priors_mm/   (R3 priors + id map)
#   data/r3_map.csv                                             (R3 n_o, p-value)
#   data/mm_combined.rds   (from 2_prior_aggregation.R)
#   data/shrinkage_feats.rds           (from 1_evidence_shrinkage.R)
# OUTPUT: output/r3_prior_shrinkage_predictor.csv  (claim_id, prior_3fam, p_base, pred)
#         — this is the deployed m2 column consumed by pipeline/10_submission.R.
#           A copy of this file is shipped in output/ so m2 is available without
#           re-running (full re-run needs the challenge-derived inputs above).
# Run from the package root.
# =============================================================================

suppressPackageStartupMessages(library(jsonlite))
cmap <- read.csv("data/r3_claimmap.csv", stringsAsFactors=FALSE)   # rnum, claim_id, type, under
cmap <- cmap[order(cmap$rnum),]; cmap$idx <- seq_len(nrow(cmap))

# --- R3 priors: Opus (rnum) + GPT/Gemini (idx) ---
opus <- NULL
for(p in c("bayesian","field","generalist","skeptic")){
  j <- fromJSON(sprintf("data/r3_priors/%s.json",p))[,c("rnum","prob")]; names(j)<-c("rnum",paste0("op_",p))
  opus <- if(is.null(opus)) j else merge(opus,j,by="rnum") }
mm <- NULL
for(m in c("gpt55","gem31")) for(p in c("bayesian","field","generalist","skeptic")){
  j <- fromJSON(sprintf("data/r3_priors_mm/%s_%s.json",m,p))[,c("idx","prob")]; names(j)<-c("idx",paste0(m,"_",p))
  mm <- if(is.null(mm)) j else merge(mm,j,by="idx") }
d <- merge(cmap, opus, by="rnum"); d <- merge(d, mm, by="idx")
pcols <- c(paste0("op_",c("bayesian","field","generalist","skeptic")),
           paste0("gpt55_",c("bayesian","field","generalist","skeptic")),
           paste0("gem31_",c("bayesian","field","generalist","skeptic")))
d$prior_3fam <- rowMeans(d[,pcols])

# --- R3 shrinkage p_base (frozen tau2 from non-health fit) ---
rm <- read.csv("data/r3_map.csv", stringsAsFactors=FALSE)
d <- merge(d, rm[,c("claim_id","n_o","pval_value_o")], by="claim_id")
tau2 <- 0.019454; Z80 <- 2.80; ZC <- 1.96
zc_from_p <- function(p) pmax(qnorm(1 - pmin(pmax(p,1e-6),0.999)/2), 1.0)
d$z <- ifelse(is.na(d$pval_value_o), NA, zc_from_p(d$pval_value_o))
d$n <- pmin(pmax(d$n_o,5),1e6)
g <- d$n*tau2/(d$n*tau2+1)
d$p_base <- pmin(pmax(pnorm(Z80*g - ZC),1e-4),1-1e-4)
d$p_base[is.na(d$p_base)] <- median(d$p_base, na.rm=TRUE)

# --- frozen combination: fit logistic(y ~ prior_3fam + p_base) on 107 labelled health ---
lab <- readRDS("data/mm_combined.rds")[,c("claim_id","y","prior_3fam","n_o")]
sf  <- readRDS("data/shrinkage_feats.rds")[,c("claim_id","n")]
lab <- merge(lab, sf, by="claim_id")
gl  <- lab$n*tau2/(lab$n*tau2+1); lab$p_base <- pmin(pmax(pnorm(Z80*gl-ZC),1e-4),1-1e-4)
fit <- glm(y ~ prior_3fam + p_base, data=lab, family=binomial)
cat("frozen combination coefficients (fit on 107 labelled health):\n"); print(round(coef(fit),3))

d$pred <- predict(fit, newdata=d, type="response")
# also prior-only and prior+logn frozen variants for comparison
fit_p  <- glm(y ~ prior_3fam, data=lab, family=binomial); d$pred_prior <- predict(fit_p,d,type="response")
lab$logn <- log(lab$n); d$logn <- log(d$n)
fit_pl <- glm(y ~ prior_3fam + logn, data=lab, family=binomial); d$pred_priorlogn <- predict(fit_pl,d,type="response")

out <- d[,c("claim_id","prior_3fam","p_base","pred","pred_prior","pred_priorlogn")]
out <- out[match(cmap$claim_id, out$claim_id),]
invisible(NULL)
write.csv(out[, c("claim_id","prior_3fam","p_base","pred")], here::here("output/r3_prior_shrinkage_predictor.csv"), row.names=FALSE)
cat(sprintf("\nR3 predictor built for %d claims.\n", nrow(out)))
cat(sprintf("pred (prior+shrinkage): mean %.3f range [%.3f, %.3f]\n", mean(out$pred), min(out$pred), max(out$pred)))
print(summary(out$pred))
