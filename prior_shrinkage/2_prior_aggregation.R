# =============================================================================
# prior_shrinkage / 2_prior_aggregation.R
#
# The PRIOR term of m2. Aggregates the LLM substantive-plausibility priors
# (4 personas x 3 model families: Claude Opus, GPT-5.5, Gemini 3.1 Pro) into a
# single `prior_3fam` per claim on the 107 labelled health rows, checks
# inter-family agreement, and reports how well the prior (alone / + log n)
# predicts replication. Confirms the prior is a distinct signal from the m3
# crowd (judge_llm_ensemble/ensemble_scores.csv).
#
# INPUTS (challenge-derived where they carry labels/sample sizes; NOT shipped):
#   data/mm_map.csv, data/poc_claim_map.csv, data/val2_claimmap.csv   (id maps + label y)
#   data/mm_priors/, data/poc_priors/, data/val2_priors/              (elicited priors, from elicit_priors.py)
#   data/poc_combined.rds, data/val2_map.csv                          (n_o per claim)
#   judge_llm_ensemble/ensemble_scores.csv                            (m3 crowd, for the comparison)
# OUTPUT: data/mm_combined.rds   (labelled training frame for step 3)
# Run from the package root.
# =============================================================================

suppressPackageStartupMessages({library(jsonlite); library(readr)})
mm <- read.csv("data/mm_map.csv", stringsAsFactors=FALSE)   # idx, claim_id, set, y

# --- multi-model priors (idx-indexed) ---
load_mm <- function(model){
  ps <- c("bayesian","field","generalist","skeptic")
  M <- NULL
  for(p in ps){ a <- fromJSON(sprintf("data/mm_priors/%s_%s.json",model,p))
    a <- a[,c("idx","prob")]; names(a) <- c("idx",paste0(model,"_",p)); M <- if(is.null(M)) a else merge(M,a,by="idx") }
  M[[paste0("prior_",model)]] <- rowMeans(M[,paste0(model,"_",ps)]); M[,c("idx",paste0("prior_",model))]
}
gpt <- load_mm("gpt55"); gem <- load_mm("gem31")

# --- Opus priors (by claim_id): 47 from scratch_poc_priors, 60 from scratch_val2_priors ---
opus47 <- { cm <- read.csv("data/poc_claim_map.csv",stringsAsFactors=FALSE)
  P<-NULL; for(p in c("bayesian","field","generalist","skeptic")){j<-fromJSON(sprintf("data/poc_priors/%s.json",p))[,c("cnum","prob")];names(j)<-c("cnum",p);P<-if(is.null(P))j else merge(P,j,by="cnum")}
  P$prior_opus<-rowMeans(P[,c("bayesian","field","generalist","skeptic")]); merge(cm[,c("cnum","claim_id")],P[,c("cnum","prior_opus")],by="cnum")[,c("claim_id","prior_opus")] }
opus60 <- { cm <- read.csv("data/val2_claimmap.csv",stringsAsFactors=FALSE)
  P<-NULL; for(p in c("bayesian","field","generalist","skeptic")){j<-fromJSON(sprintf("data/val2_priors/%s.json",p))[,c("vnum","prob")];names(j)<-c("vnum",p);P<-if(is.null(P))j else merge(P,j,by="vnum")}
  P$prior_opus<-rowMeans(P[,c("bayesian","field","generalist","skeptic")]); merge(cm[,c("vnum","claim_id")],P[,c("vnum","prior_opus")],by="vnum")[,c("claim_id","prior_opus")] }
opus <- rbind(opus47, opus60)

d <- merge(mm, gpt, by="idx"); d <- merge(d, gem, by="idx"); d <- merge(d, opus, by="claim_id")
# n_o
n47 <- readRDS("data/poc_combined.rds")[,c("claim_id","n_o")]
n60 <- read.csv("data/val2_map.csv",stringsAsFactors=FALSE)[,c("claim_id","n_o")]
d <- merge(d, rbind(n47,n60), by="claim_id")
d$logn <- log(pmin(pmax(d$n_o,5),1e6))
d$prior_3fam <- rowMeans(d[,c("prior_opus","prior_gpt55","prior_gem31")])

auc<-function(y,s){ok<-!is.na(s)&!is.na(y);y<-y[ok];s<-s[ok];if(length(unique(y))<2)return(NA);r<-rank(s);(sum(r[y==1])-sum(y)*(sum(y)+1)/2)/(sum(y)*sum(1-y))}
brier<-function(y,p)mean((p-y)^2)
loo<-function(f,data){data<-data[complete.cases(data[,all.vars(f)]),];p<-numeric(nrow(data));for(i in seq_len(nrow(data))){m<-suppressWarnings(glm(f,data=data[-i,],family=binomial));p[i]<-predict(m,newdata=data[i,],type="response")};brier(data$y,p)}

cat("=== inter-family correlations (n=107) ===\n")
cat(sprintf("opus-gpt %.2f | opus-gem %.2f | gpt-gem %.2f\n",
  cor(d$prior_opus,d$prior_gpt55), cor(d$prior_opus,d$prior_gem31), cor(d$prior_gpt55,d$prior_gem31)))

for(lab in list(c("v1_47","R3-analog (R1/R2 health)"), c("v1_47,v2_60","ALL health 107"))){
  sub <- if(grepl(",",lab[1])) d else d[d$set==lab[1],]
  cat(sprintf("\n=== %s (n=%d, base %.3f) ===\n", lab[2], nrow(sub), mean(sub$y)))
  for(v in c("prior_opus","prior_gpt55","prior_gem31","prior_3fam")) cat(sprintf("  AUC %-12s %.3f\n", v, auc(sub$y, sub[[v]])))
  cat(sprintf("  LOO Brier: 3fam %.4f | 3fam+logn %.4f | opus %.4f | opus+logn %.4f | base %.4f\n",
    loo(y~prior_3fam,sub), loo(y~prior_3fam+logn,sub), loo(y~prior_opus,sub), loo(y~prior_opus+logn,sub), brier(sub$y,rep(mean(sub$y),nrow(sub)))))
}

# correlation of 3-family prior with the existing ('pure LLM') ensemble, on the 47
ens <- read_csv("judge_llm_ensemble/ensemble_scores.csv", show_col_types=FALSE)[,c("claim_id","p_replication_observed")]
de <- merge(d[d$set=="v1_47",], ens, by="claim_id")
cat(sprintf("\n=== vs existing 'pure LLM' ensemble (n=%d) ===\n", nrow(de)))
cat(sprintf("corr(3fam prior, ensemble) = %.2f (Pearson), %.2f (Spearman)\n",
  cor(de$prior_3fam, de$p_replication_observed), cor(de$prior_3fam, de$p_replication_observed, method="spearman")))
cat(sprintf("AUC ensemble %.3f | 3fam prior %.3f\n", auc(de$y,de$p_replication_observed), auc(de$y,de$prior_3fam)))
saveRDS(d, "data/mm_combined.rds")
