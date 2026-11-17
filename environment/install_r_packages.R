# Installs the R packages that run_all.sh needs beyond base R and reports their versions.
# Missing packages are installed from CRAN at their current versions; a version that differs from
# the tested one gets a warning (results were checked only with the tested versions).
tested <- c(data.table = "1.18.2.1", haven = "2.5.5", jsonlite = "2.0.0", sandwich = "3.1-1")
missing <- setdiff(names(tested), rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
cat(R.version.string, "(tested with R 4.5.1)\n")
for (p in names(tested)) {
  have <- packageVersion(p)
  cat(sprintf("%-10s %-10s (tested with %s)\n", p, as.character(have), tested[[p]]))
  if (have != package_version(tested[[p]]))
    warning(sprintf("%s %s differs from the tested version %s", p, have, tested[[p]]), call. = FALSE)
}
