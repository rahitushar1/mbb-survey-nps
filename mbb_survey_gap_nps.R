# ============================================================
# Maryland MBB Survey — Clean Script
#   - Data overview + NPS summaries
#   - Experience gaps (IMP vs SAT)
#   - NPS drivers (3 approaches):
#       (1) Composite OLS
#       (2) LASSO (mean-impute) -> post-LASSO OLS
#       (3) RF impute -> LASSO -> post-LASSO OLS
#   - Next season intent (BINOMIAL only) + LASSO selection
# ============================================================

# ---- 0) Libraries ----
library(dplyr)
library(glmnet)
library(missForest)
library(DescTools)

# ---- 1) Load data ----
data_path <- file.path("data", "R1.csv")
if (!file.exists(data_path)) stop("Missing data/R1.csv (not included in repo). Put it locally inside data/ to run.")
data <- read.csv(data_path, header = TRUE)

# ---- 2) Fix unnamed SAT columns (your 3 ENG items) ----
names(data)[names(data) == "Unnamed.6"] <- "ENG_SAT_cheers"
names(data)[names(data) == "Unnamed.7"] <- "ENG_SAT_halftime"
names(data)[names(data) == "Unnamed.8"] <- "ENG_SAT_host"

# Normalize "Dissatisfied" label for those 3 columns
fix_dissatisfied <- function(x) {
  x[x == "Dissatisfied"] <- "Somewhat Dissatisfied"
  x
}
data$ENG_SAT_cheers   <- fix_dissatisfied(data$ENG_SAT_cheers)
data$ENG_SAT_halftime <- fix_dissatisfied(data$ENG_SAT_halftime)
data$ENG_SAT_host     <- fix_dissatisfied(data$ENG_SAT_host)

# ---- 3) Identify blocks (after renaming) ----
sat_cols   <- grep("_SAT_",   names(data), value = TRUE)
imp_cols   <- grep("_IMP_",   names(data), value = TRUE)
atpac_cols <- grep("^AT_PAC_", names(data), value = TRUE)

# ---- 4) Recode maps ----
recode_from_map <- function(x, map) {
  out <- unname(map[as.character(x)])
  as.numeric(out)
}

sat_map <- c("Very Dissatisfied" = 1, "Somewhat Dissatisfied" = 2, "Satisfied" = 3, "Very Satisfied" = 4)
imp_map <- c("Not Important" = 1, "Somewhat Important" = 2, "Very Important" = 3, "Critical" = 4)
atpac_map <- c("No Impact" = 1, "Little Impact" = 2, "Some impact" = 3, "Significant impact" = 4)

data[sat_cols]   <- lapply(data[sat_cols],   recode_from_map, map = sat_map)
data[imp_cols]   <- lapply(data[imp_cols],   recode_from_map, map = imp_map)
data[atpac_cols] <- lapply(data[atpac_cols], recode_from_map, map = atpac_map)

# ---- 5) Feature engineering (ONLY what you actually use) ----
# Games attended midpoint
games_mid <- c("0"=0, "1-3"=2, "4-8"=6, "9-13"=11, "14 or more"=16)
data$Games_attended_mid <- games_mid[as.character(data$Games_attended)]

# Travel midpoint
travel_mid <- c("Less than 30 minutes"=15,"30 minutes - 1 hour"=45,"1-2 hours"=90,"3-4 hours"=210,"More than 4 hours"=300)
data$travelmid <- travel_mid[as.character(data$Travel_distance)]

# Seating premium flag
data$Seat_is_premium <- as.integer(grepl("Premium Seating", data$Seating_type))

# Customer flags used in your slides/segment NPS
data$ISstudent      <- as.integer(grepl("University of Maryland Student", data$Customer_type))
data$ISseasonholder <- as.integer(grepl("Maryland Men's Basketball Season Ticket Holder", data$Customer_type))
data$ISsinglegame   <- as.integer(grepl("Single Game Buyer", data$Customer_type))

# ---- 6) Composite SAT scores for Composite OLS approach ----
row_mean_min_k <- function(df_cols, k = 2, na.rm = TRUE) {
  nonmiss <- rowSums(!is.na(df_cols))
  x <- rowMeans(df_cols, na.rm = na.rm)
  x[is.nan(x)] <- NA
  x[nonmiss < k] <- NA
  x
}

data$gexsat_parkingtraffic <- row_mean_min_k(data[, c("GEX_SAT_Parking","GEX_SAT_pretraffic","GEX_SAT_posttraffic")])
data$gexsat_seating        <- row_mean_min_k(data[, c("GEX_SAT_seatmark","GEX_SAT_seatcomf","GEX_SAT_seatloc")])
data$gexsat_arena          <- row_mean_min_k(data[, c("GEX_SAT_gateclean","GEX_SAT_outappear","GEX_SAT_inappear")])
data$gexsat_entrystaff     <- row_mean_min_k(data[, c("GEX_SAT_tickettakespeed","GEX_SAT_tickettakesfriend","GEX_SAT_secprof")])

data$catsat_fbqual   <- row_mean_min_k(data[, c("CAT_SAT_foodqual","CAT_SAT_foodsel","CAT_SAT_bevqual","CAT_SAT_bevsel")])
data$catsat_fbprice  <- row_mean_min_k(data[, c("CAT_SAT_foodprice","CAT_SAT_bevprice")])
data$catsat_service  <- row_mean_min_k(data[, c("CAT_SAT_staffperf","CAT_SAT_stafffriend","CAT_SAT_wait")])

data$rstsat_comb <- row_mean_min_k(data[, c("RST_SAT_waitlen","RST_SAT_clean","RST_SAT_qual")])

data$vidsat_info  <- row_mean_min_k(data[, c("VID_SAT_otherscore","VID_SAT_teamstat","VID_SAT_highlight")])
data$vidsat_music <- row_mean_min_k(data[, c("VID_SAT_musiclar","VID_SAT_musicvol","VID_SAT_musicsel","VID_SAT_overallsound")])
data$vidsat_pa    <- row_mean_min_k(data[, c("VID_SAT_paclar","VID_SAT_pavol","VID_SAT_pagameact")])
data$vidsat_video <- row_mean_min_k(data[, c("VID_SAT_pic","VID_SAT_content","VID_SAT_ribbcontent")])

data$mobsat_comb <- row_mean_min_k(data[, c("MOB_SAT_txt","MOB_SAT_call","MOB_SAT_social","MOB_SAT_stream")])

# ============================================================
# A) DESCRIPTIVES (used in deck)
# ============================================================
prop.table(table(na.exclude(data$ISstudent)))
prop.table(table(na.exclude(data$ISseasonholder)))
prop.table(table(na.exclude(data$ISsinglegame)))
prop.table(table(na.exclude(data$Gender)))
prop.table(table(na.exclude(data$Age_group)))
prop.table(table(na.exclude(data$Travel_distance)))
prop.table(table(na.exclude(data$Games_attended)))

mean(data$NPS, na.rm = TRUE)
sd(data$NPS, na.rm = TRUE)

aggregate(NPS ~ ISstudent,      data = data, mean, na.rm = TRUE)
aggregate(NPS ~ ISseasonholder, data = data, mean, na.rm = TRUE)
aggregate(NPS ~ Travel_distance,data = data, mean, na.rm = TRUE)

# ============================================================
# B) EXPERIENCE GAPS (robust SAT/IMP matching by key)
# ============================================================
sat_means <- data.frame(
  key = sub("^.*_SAT_", "", sat_cols),
  sat_col = sat_cols,
  Avg_SAT = colMeans(data[sat_cols], na.rm = TRUE)
)

imp_means <- data.frame(
  key = sub("^.*_IMP_", "", imp_cols),
  imp_col = imp_cols,
  Avg_IMP = colMeans(data[imp_cols], na.rm = TRUE)
)

gap_table <- inner_join(sat_means, imp_means, by = "key") %>%
  mutate(Gap = Avg_SAT - Avg_IMP) %>%
  arrange(Gap)

head(gap_table, 15)

# ============================================================
# C) NPS DRIVER MODELS (3 approaches)
# ============================================================

# ---- C1) Composite OLS ----
m_comp <- lm(
  NPS ~ gexsat_parkingtraffic + gexsat_entrystaff + gexsat_arena + gexsat_seating +
    catsat_fbqual + catsat_fbprice + catsat_service +
    rstsat_comb + vidsat_video + vidsat_pa + vidsat_music + vidsat_info +
    mobsat_comb + BND_SAT_overall,
  data = data
)
summary(m_comp)

# ---- helper: LASSO select ----
lasso_select <- function(x, y, family = "gaussian", pick = c("1se","min"), seed = 123) {
  pick <- match.arg(pick)
  set.seed(seed)
  cvfit <- cv.glmnet(x, y, family = family, alpha = 1, standardize = TRUE)
  s_val <- if (pick == "1se") "lambda.1se" else "lambda.min"
  co <- coef(cvfit, s = s_val)
  
  sel <- rownames(co)[co[,1] != 0]
  sel <- setdiff(sel, "(Intercept)")
  list(cvfit = cvfit, selected = sel, pick = s_val)
}

# ---- C2) LASSO on raw SAT (mean-impute) -> post-LASSO OLS ----
lasso_mean_df <- data %>%
  select(NPS, all_of(sat_cols)) %>%
  filter(!is.na(NPS))

for (v in sat_cols) {
  lasso_mean_df[[v]][is.na(lasso_mean_df[[v]])] <- mean(lasso_mean_df[[v]], na.rm = TRUE)
}

x2 <- model.matrix(NPS ~ ., data = lasso_mean_df)[, -1]
y2 <- lasso_mean_df$NPS

sel2 <- lasso_select(x2, y2, family = "gaussian", pick = "1se")
sel2$selected

m_lasso_mean <- lm(
  as.formula(paste("NPS ~", paste(sel2$selected, collapse = " + "))),
  data = lasso_mean_df
)
summary(m_lasso_mean)

# ---- C3) RF-impute SAT -> LASSO -> post-LASSO OLS ----
sat_only <- data[, sat_cols]
sat_only[] <- lapply(sat_only, as.numeric)

set.seed(123)
rf_imp <- missForest(sat_only, maxiter = 5, ntree = 100)
sat_imp <- rf_imp$ximp

lasso_rf_df <- data.frame(NPS = data$NPS, sat_imp) %>%
  filter(!is.na(NPS))

x3 <- model.matrix(NPS ~ ., data = lasso_rf_df)[, -1]
y3 <- lasso_rf_df$NPS

sel3 <- lasso_select(x3, y3, family = "gaussian", pick = "1se")  # or "min"
m_lasso_rf <- lm(
  as.formula(paste("NPS ~", paste(sel3$selected, collapse = " + "))),
  data = lasso_rf_df
)
summary(m_lasso_rf)

# Compare R^2 (note: sample sizes differ across approaches)
nps_r2 <- data.frame(
  model = c("Composite OLS", "LASSO(mean-imp)->OLS", "RF-imp->LASSO->OLS"),
  n     = c(nobs(m_comp), nobs(m_lasso_mean), nobs(m_lasso_rf)),
  r2    = c(summary(m_comp)$r.squared, summary(m_lasso_mean)$r.squared, summary(m_lasso_rf)$r.squared),
  adj_r2= c(summary(m_comp)$adj.r.squared, summary(m_lasso_mean)$adj.r.squared, summary(m_lasso_rf)$adj.r.squared)
)
nps_r2