#!/usr/bin/env Rscript
# Metadata and methods audit only. No patient records are read or exported.
# New audit software written in October 2026; not historical analysis code.

cat("OphA-PRISM R methods audit 0.1.0\n")
cat("R:", as.character(getRversion()), "\n")
packages <- c("mice", "pmsampsize", "mclust", "logistf", "irr", "boot")
for (name in packages) {
  if (requireNamespace(name, quietly = TRUE)) {
    cat(name, ":", as.character(utils::packageVersion(name)), "\n")
  } else {
    cat(name, ": unavailable; nothing was installed\n")
  }
}

cat("\nTHEORETICAL MAXIMUM COX-SNELL R-SQUARED FROM PREVALENCE\n")
for (event_count in c(23, 50)) {
  p <- event_count / 261
  maximum <- 1 - exp(2 * (p * log(p) + (1 - p) * log(1 - p)))
  cat("Events:", event_count, "; maximum Cox-Snell R-squared:", format(maximum, digits = 10), "\n")
}

if (requireNamespace("pmsampsize", quietly = TRUE)) {
  cat("\nRECALCULATION OF AUTHOR-SPECIFIED CURE SAMPLE-SIZE INPUTS\n")
  cat('pmsampsize(type="b", parameters=15, rsquared=0.598, prevalence=50/261, shrinkage=0.90)\n')
  # Use the full documented argument name; the author's rsq abbreviation refers
  # to rsquared in this package. This is a methodological calculation only.
  tryCatch({
    answer <- pmsampsize::pmsampsize(type = "b", parameters = 15,
                                   rsquared = 0.598, prevalence = 50 / 261,
                                   shrinkage = 0.90)
    print(answer)
  }, error = function(e) {
    cat("Calculation did not complete:", conditionMessage(e), "\n")
  })
}

if (requireNamespace("irr", quietly = TRUE)) {
  cat("\nFLEISS KAPPA MISSING-RATER HANDLING\n")
  source_text <- paste(deparse(body(irr::kappam.fleiss)), collapse = "\n")
  cat("Installed function contains na.omit:", grepl("na.omit", source_text, fixed = TRUE), "\n")
  # These are four artificial software-test rows, not clinical observations.
  toy <- matrix(c(1,1,1, 2,2,NA, 1,2,1, 2,2,2), nrow = 4, byrow = TRUE)
  tryCatch({
    answer <- irr::kappam.fleiss(toy)
    cat("Artificial test input rows: 4; rows used by function:", answer$subjects, "\n")
  }, error = function(e) cat("Artificial missing-rater test did not complete.\n"))
  cat("This diagnostic does not calculate or validate the study's clinician agreement.\n")
}

if (requireNamespace("mice", quietly = TRUE)) {
  cat("\nMICE API CHECK\n")
  complete_function <- getS3method("complete", "mids", envir = asNamespace("mice"))
  cat("complete.mids formal arguments:", paste(names(formals(complete_function)), collapse = ", "), "\n")
  cat("Actual held-out imputation requires inspection of mice(), predictorMatrix, ignore, where, and outcome masking.\n")
}
