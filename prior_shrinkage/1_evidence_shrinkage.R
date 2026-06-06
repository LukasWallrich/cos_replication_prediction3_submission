# =============================================================================
# prior_shrinkage / 1_evidence_shrinkage.R
#
# The EVIDENCE term of the m2 "prior x evidence" predictor.
#
# Idea: among already-published, already-significant findings the original
# p-value is uninformative (selection range-restricts it); sample size is the
# live axis. Under a replication powered for the OBSERVED effect, the original
# effect size cancels analytically and the predictive replication probability
# becomes a shrunken function of the original sample size:
#
#   d_o = z_o / sqrt(n),  n_r = (2.80 / d_o)^2,  g(n) = n*tau2 / (n*tau2 + 1)
#   base:  p_rep = Phi(2.80 * g(n) - 1.96)
#   het:   p_rep = Phi((2.80 * g(n) - 1.96) / sqrt(1 + n_r * tauh2))
#
# tau2 (and the heterogeneity tauh2) are fit on NON-HEALTH rows only, then
# frozen and evaluated on health — so the form is never tuned on the target
# domain. The frozen tau2 (~0.0195) is what 3_build_predictor.R uses for R3.
#
# INPUT : data/train_base.rds        (challenge-derived; NOT shipped — see DATA.md)
# OUTPUT: data/shrinkage_feats.rds   (claim_id, n, nr, z, d)  [challenge-derived]
# Run from the package root.
# =============================================================================

suppressPackageStartupMessages(library(dplyr))
tb <- readRDS("data/train_base.rds")
gt <- tb %>% filter(!is.na(statistical_success), dataset %in% c("training","round1","round2"),
                    !is.na(n_o), !is.na(pval_value_o))
gt$y  <- as.integer(gt$statistical_success)
gt$hb <- ifelse(is.na(gt$health_related_big),0L,as.integer(gt$health_related_big==1|gt$health_related_big==TRUE))
gt$n  <- pmin(pmax(gt$n_o,5),1e6)
gt$z  <- qnorm(1 - pmin(pmax(gt$pval_value_o,1e-6),0.999)/2)
gt$z  <- pmax(gt$z, 1.0)               # floor (published) to keep d_o finite
gt$d  <- gt$z/sqrt(gt$n)
gt$nr <- (2.80/gt$d)^2
gt$logn <- log(gt$n)
Z80 <- 2.80; ZC <- 1.96
clip <- function(p) pmin(pmax(p,1e-4),1-1e-4)

p_base <- function(n, lt2){ tau2<-exp(lt2); g<-n*tau2/(n*tau2+1); clip(pnorm(Z80*g - ZC)) }
p_het  <- function(n, nr, par){ tau2<-exp(par[1]); th2<-exp(par[2]); g<-n*tau2/(n*tau2+1)
                                clip(pnorm((Z80*g - ZC)/sqrt(1 + nr*th2))) }
nll_base <- function(lt2,d) { p<-p_base(d$n,lt2); -sum(d$y*log(p)+(1-d$y)*log(1-p)) }
nll_het  <- function(par,d) { p<-p_het(d$n,d$nr,par); -sum(d$y*log(p)+(1-d$y)*log(1-p)) }
brier<-function(y,p)mean((p-y)^2)
auc<-function(y,s){r<-rank(s);(sum(r[y==1])-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(1-y))}

NH <- gt %>% filter(hb==0); H <- gt %>% filter(hb==1)
cat(sprintf("non-health %d | health %d | health base %.3f\n", nrow(NH), nrow(H), mean(H$y)))

# fit on NON-HEALTH
fb <- optimize(nll_base, c(-12,2), d=NH)
fh <- optim(c(fb$minimum,-8), nll_het, d=NH, method="Nelder-Mead")
cat(sprintf("\nfitted on non-health: base tau2=%.5g | het tau2=%.5g tauh2=%.5g\n",
            exp(fb$minimum), exp(fh$par[1]), exp(fh$par[2])))

# freeze -> health
H$pb <- p_base(H$n, fb$minimum)
H$ph <- p_het(H$n, H$nr, fh$par)
# frozen logistic(logn) baseline: fit on NH, predict H
glm_ln <- glm(y~logn, data=NH, family=binomial); H$pln <- predict(glm_ln, H, type="response")
cat(sprintf("\n=== FROZEN (fit non-health) -> HEALTH (n=%d) ===\n", nrow(H)))
cat(sprintf("  base rate            Brier %.4f\n", brier(H$y, rep(mean(NH$y),nrow(H)))))
cat(sprintf("  logistic(log n)      Brier %.4f  AUC %.3f\n", brier(H$y,H$pln), auc(H$y,H$pln)))
cat(sprintf("  shrinkage base       Brier %.4f  AUC %.3f\n", brier(H$y,H$pb),  auc(H$y,H$pb)))
cat(sprintf("  shrinkage + het      Brier %.4f  AUC %.3f\n", brier(H$y,H$ph),  auc(H$y,H$ph)))
cat(sprintf("\n  (sanity) predicted p_rep range base [%.2f,%.2f], het [%.2f,%.2f]\n",
            min(H$pb),max(H$pb),min(H$ph),max(H$ph)))

# challenge-health (R3 analog) subset
HC <- H %>% filter(dataset %in% c("round1","round2"))
cat(sprintf("\n=== R3-analog challenge-health (n=%d, base %.3f) ===\n", nrow(HC), mean(HC$y)))
cat(sprintf("  logistic(logn) Brier %.4f | shrink-base %.4f | shrink-het %.4f\n",
            brier(HC$y,HC$pln), brier(HC$y,HC$pb), brier(HC$y,HC$ph)))
saveRDS(gt %>% transmute(claim_id, n, nr, z, d), "data/shrinkage_feats.rds")
