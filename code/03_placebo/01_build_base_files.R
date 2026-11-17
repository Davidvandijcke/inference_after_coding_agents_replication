# Base file of each placebo study: real baseline variables and outcomes under neutral names and a randomization
# block, with no treatment, identifier, date or open-text column. The teachers base file is copied from the
# shipped extract.
#
# Inputs:  data/raw/jones_molitor_reif_2019/online_surveys.csv, data/raw/kaufmann_andre_koszegi_2024/wide.dta,
#          data/raw/costa_ramon_schaede_slotwinski_2026/teachers_base.csv, teachers_design.json
# Outputs: data/derived/03_placebo/base/{wellness,teachers,consumers}_base.csv, experiments.json
suppressMessages({library(data.table); library(haven); library(jsonlite)})
args <- commandArgs(FALSE)
# Rscript writes spaces in the script path as "~+~".
here <- dirname(normalizePath(gsub("~+~", " ", sub("^--file=", "", args[grep("^--file=", args)]), fixed = TRUE)))
root <- normalizePath(file.path(here, "..", ".."))
raw <- file.path(root, "data", "raw")
out <- file.path(root, "data", "derived", "03_placebo", "base")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

num <- function(x) { x <- as.numeric(x); ifelse(is.na(x), NA_character_, as.character(round(x, 3))) }

# Treated count per block: the study's treated total allocated across blocks by largest remainder.
allocate <- function(block, treated_total) {
  n_b <- table(block); share <- treated_total / sum(n_b)
  k <- floor(n_b * share); rem <- n_b * share - k
  extra <- treated_total - sum(k)
  if (extra > 0) { up <- order(-as.numeric(rem))[seq_len(extra)]; k[up] <- k[up] + 1 }
  as.list(setNames(as.integer(k), names(n_b)))
}

finish <- function(name, d, block, treated_total) {
  fwrite(cbind(data.table(block = as.integer(block)), d), file.path(out, paste0(name, "_base.csv")),
         na = "", quote = FALSE)
  list(n = nrow(d), treated_total = treated_total, treated_per_block = allocate(block, treated_total))
}

## Wellness: blocks are sex x age group x race.
w <- fread(file.path(raw, "jones_molitor_reif_2019", "online_surveys.csv"), na.strings = "", showProgress = FALSE)
stems <- c(everscreen = "ever_screened", active = "physically_active", active_try = "trying_more_active",
           cursmk = "current_smoker", othersmk = "other_tobacco", formsmk = "former_smoker", drink = "drinker",
           drinkhvy = "heavy_drinker", chronic = "chronic_condition", health1 = "health_very_good",
           health2 = "health_not_poor", problems = "physical_problems", energy = "lots_of_energy",
           ehealth = "bad_emotional_health", overweight = "overweight", badhealth = "high_bp_chol_glucose",
           sedentary = "sedentary_job", druguse = "takes_medication", physician = "physician_or_er_visit",
           hospital = "hospital_stay", sickdays = "any_sick_days", hrsworked50 = "works_50_plus_hours",
           jobsatisf1 = "very_satisfied_job", jobsatisf2 = "satisfied_job", mgmtsafety = "mgmt_values_safety")
later <- c(happywork = "happier_at_work", presenteeism = "presenteeism_score", productive = "very_productive",
           promotion = "promoted", jobsearch1 = "job_search_very_likely", jobsearch2 = "job_search_likely")
wd <- data.table(male = w$male, age_group = fifelse(w$age50 == 1, 3L, fifelse(w$age37_49 == 1, 2L, 1L)), white = w$white)
stopifnot(!anyNA(wd$male), !anyNA(wd$age_group), !anyNA(wd$white))  # paste() below would turn NA into a block
for (s in names(stems)) wd[[paste0(stems[s], "_base")]] <- w[[paste0(s, "_0716")]]
for (s in names(stems)) {
  wd[[paste0(stems[s], "_y1")]] <- w[[paste0(s, "_0717")]]
  wd[[paste0(stems[s], "_y2")]] <- w[[paste0(s, "_0718")]]
}
for (s in names(later)) {
  wd[[paste0(later[s], "_y1")]] <- w[[paste0(s, "_0717")]]
  wd[[paste0(later[s], "_y2")]] <- w[[paste0(s, "_0718")]]
}
meta_w <- finish("wellness", wd, as.integer(factor(paste(wd$male, wd$age_group, wd$white))), sum(w$treat))

## Teachers: the shipped extract and the study's treated total.
teachers <- file.path(raw, "costa_ramon_schaede_slotwinski_2026")
stopifnot(file.copy(file.path(teachers, "teachers_base.csv"), file.path(out, "teachers_base.csv"), overwrite = TRUE))
tb <- fread(file.path(out, "teachers_base.csv"), select = "block")$block
td <- fromJSON(file.path(teachers, "teachers_design.json"))
stopifnot(!anyNA(tb), length(tb) == td$n)
meta_t <- list(n = td$n, treated_total = td$treated_total, treated_per_block = allocate(tb, td$treated_total))

## Consumers: blocks are the eight assigned consumption topics; half of the respondents get the invented module.
g <- as.data.table(read_dta(file.path(raw, "kaufmann_andre_koszegi_2024", "wide.dta")))
scen <- c(fuel = 1L, meat = 2L, flights = 3L, energy = 4L, subelectricity = 5L, subhouse = 6L, subclothing = 7L, subcoffee = 8L)
topic <- c(co2 = "co2", waste = "waste", chicken = "animal_welfare", textile = "low_wages")
cd <- data.table(
  wtp_topic = unname(topic[g$conseq]),
  gender = fifelse(as.numeric(g$gender) %in% c(1, 2), as.character(as.numeric(g$gender)), "3"),
  age = num(g$age), region = g$region, education = num(g$education), income = num(g$income), politics = num(g$politics),
  race_white = num(g$racewhite_yes), race_black = num(g$raceblack_yes), race_hispanic = num(g$racehispanic_yes),
  race_asian = num(g$raceasian_yes), race_indigenous = num(g$raceindigenous_yes),
  uses_topic_good = num(g$consumer_vig), uses_wtp_good = num(g$consumer_conseq),
  effect_category = num(g$effect), effect_below_one_to_one = num(g$belief_dampening), effect_per_unit = num(g$quant_effect),
  reason_type = c(naive = "direct", dampening = "market_offset", spillover = "spillover", other = "other")[g$code_effect_simple],
  reason_market_offsets = num(g$explanation_dampening),
  wtp_effective = num(g$wtp_1), wtp_ineffective = num(g$wtp_2),
  wtp_effective_capped = num(g$wtp_1_censored), wtp_ineffective_capped = num(g$wtp_2_censored),
  wtp_ratio = num(g$wtp_ratio), wtp_zero_if_ineffective = num(g$conseq_strict), wtp_less_if_ineffective = num(g$conseq_weak),
  wtp_equal_either_way = num(g$conseq_deont), wtp_zero_either_way = num(g$conseq_dontcare))
cb <- unname(scen[g$vig])
stopifnot(!anyNA(cb), !anyNA(cd$wtp_topic))
meta_c <- finish("consumers", cd, cb, nrow(cd) %/% 2L)

write_json(list(wellness = meta_w, teachers = meta_t, consumers = meta_c), file.path(out, "experiments.json"),
           auto_unbox = TRUE, pretty = TRUE)
for (e in list(c("wellness", meta_w$n), c("teachers", meta_t$n), c("consumers", meta_c$n))) cat(e[1], e[2], "rows\n")
