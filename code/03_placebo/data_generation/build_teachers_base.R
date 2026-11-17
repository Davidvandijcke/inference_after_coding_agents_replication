# Regenerates the shipped teachers base file from the authors' replication package (not run by run.sh).
#
# The source file, 3_clean_fu1.dta in Costa-Ramón, Slotwinski, Schaede and Brenøe's Harvard Dataverse deposit
# (doi:10.7910/DVN/A9ZFFF), is not redistributed; the package ships the numeric extract this script writes
# (data/raw/costa_ramon_schaede_slotwinski_2026/SOURCE.md). The script reads from the file only the 70
# survey columns listed below, which it writes under the names the agents saw, and the real assignment
# (treat_all), which it uses only to count the treated participants. It writes no identifier, string or
# treatment column.
#
# Input:   QJE_replicationpackage_transfer.zip, downloaded from the Dataverse record (checked by SHA-256)
# Outputs: OUT_DIR/teachers_base.csv and OUT_DIR/teachers_design.json, which should equal the files shipped in
#          data/raw/costa_ramon_schaede_slotwinski_2026/ byte for byte.
# Usage:   Rscript build_teachers_base.R QJE_replicationpackage_transfer.zip OUT_DIR
suppressMessages({library(data.table); library(haven); library(jsonlite)})
args <- commandArgs(TRUE)
zip <- args[1]; out <- args[2]
stopifnot(unname(tools::sha256sum(zip)) == "7a500b4d0dd6973d758b73a27a4d17e3fe92df91aaf44f0f43634960a117ac07")
member <- "QJE_replicationpackage_transfer/data/2_clean/1_intermediate/3_clean_fu1.dta"
tmp <- tempfile(); dir.create(tmp)
unzip(zip, files = member, exdir = tmp, junkpaths = TRUE)
dta <- file.path(tmp, basename(member))
stopifnot(unname(tools::sha256sum(dta)) == "f4e3833abe9224da9b01255139d37f70144798c2aefc46b0787d9a04ed7abc1f")

num <- function(x) { x <- as.numeric(x); ifelse(is.na(x), NA_character_, as.character(round(x, 3))) }

# Source column (authors' name) = column the agents saw; meanings are in data_generation/runtime/codebooks/teachers.md.
keep <- c(age_group = "age_group", number_children = "number_children", age_child = "age_youngest_child",
          marital_status = "partner_status", fulltime = "works_full_time", wl_current = "employment_pct_base",
          teacher = "classroom_teacher", teaching_diploma = "teaching_diploma", wl_1y_pre = "plan_employment_1y_base",
          org_fam = "ease_increase_family", org_work = "ease_increase_employer", pension_estimate = "pension_estimate",
          childcare_cost = "childcare_cost_view", suffer_child = "child_suffers_view",
          suffer_famlife = "family_suffers_view", father_able = "father_able_view",
          external_childcare = "external_care_view",
          wl_1y = "plan_employment_1y_post", wl_10y = "plan_employment_10y_post", voucher = "chose_consultation_post",
          wl_trust = "if_trusted_care_post", wl_norms = "if_no_norms_post", wl_pay = "if_higher_pay_post",
          wl_flex = "if_partner_flexible_post", wl_engaged = "if_partner_engaged_post",
          advice_childcare_w1 = "rank_childcare_post", advice_salary_w1 = "rank_salary_post",
          advice_pension_w1 = "rank_pension_post", advice_progress_w1 = "rank_career_post",
          care_correct = "correct_ranking_post",
          f_angry = "felt_angry_post", f_anxious = "felt_anxious_post", f_hopeful = "felt_hopeful_post",
          f_discouraged = "felt_discouraged_post", f_happy = "felt_happy_post", f_motivated = "felt_motivated_post",
          info_ahv = "wants_pension_overview_post", info_relationships = "wants_couple_finance_post",
          info_security = "wants_security_info_post", info_pension_gaps = "wants_pension_course_post",
          info_tool = "wants_online_tool_post", info_health = "wants_premium_info_post",
          wl_1y_fu = "plan_employment_1y_fu", wl_10y_fu = "plan_employment_10y_fu",
          inc_wl_3y = "predicted_employment_3y_fu", inc_wl_5y = "predicted_employment_5y_fu",
          inc_wl_10y = "predicted_employment_10y_fu", p_wl_1y = "partner_employment_1y_fu",
          sat_famtime = "satisfied_family_time_fu", sat_partner = "satisfied_relationship_fu",
          sat_comprehension = "feels_understood_fu", sat_division = "satisfied_chores_fu", sat_purpose = "job_purpose_fu",
          implement_wl = "implemented_level_fu", timing_decision = "decision_timing_fu",
          discuss_partner = "discussed_partner_fu", discuss_colleague = "discussed_colleagues_fu",
          discuss_friends = "discussed_friends_fu", discuss_any = "discussed_anyone_fu", action_fu1 = "took_action_fu",
          f_nervous = "stressed_fu", f_myway = "things_going_well_fu", f_accomplished = "cannot_cope_fu", f_control = "in_control_fu",
          more_children = "wants_more_children_fu",
          advice_childcare_fu1 = "rank_childcare_fu", advice_salary_fu1 = "rank_salary_fu",
          advice_pension_fu1 = "rank_pension_fu", advice_progress_fu1 = "rank_career_fu",
          care_correct_fu1 = "correct_ranking_fu")
# Read these columns and the real assignment, and nothing else, from the file. Keep the randomized survey
# sample with an observed age group (the randomization block).
m <- as.data.table(read_dta(dta, col_select = tidyselect::all_of(c("treat_all", names(keep)))))
unlink(tmp, recursive = TRUE)
m <- m[!is.na(treat_all) & !is.na(age_group)]
d <- as.data.table(lapply(names(keep), function(v) num(m[[v]])))
setnames(d, unname(keep))
dir.create(out, showWarnings = FALSE, recursive = TRUE)
fwrite(cbind(data.table(block = as.integer(m$age_group)), d), file.path(out, "teachers_base.csv"), na = "", quote = FALSE)
write_json(list(n = nrow(d), treated_total = sum(m$treat_all == 1)), file.path(out, "teachers_design.json"),
           auto_unbox = TRUE, pretty = TRUE)
cat("teachers:", nrow(d), "rows,", ncol(d), "survey columns,", sum(m$treat_all == 1), "treated\n")
