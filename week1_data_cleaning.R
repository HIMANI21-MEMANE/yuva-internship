# Week 1: Data Cleaning and Preliminary Analysis with R
# Dataset: Titanic passenger dataset (891 rows, 12 columns)
# Public source: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

# Install once if needed:
# install.packages(c("ggplot2", "dplyr"))

library(ggplot2)
library(dplyr)

url <- "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_raw <- read.csv(url, stringsAsFactors = FALSE, na.strings = c("", "NA"))

# 1. Initial exploration
cat("Dimensions (rows, columns):\\n")
print(dim(titanic_raw))
cat("\\nStructure:\\n")
str(titanic_raw)
cat("\\nSummary:\\n")
print(summary(titanic_raw))
cat("\\nFirst six rows:\\n")
print(head(titanic_raw))

# 2. Missing-value audit
missing_count <- sapply(titanic_raw, function(x) sum(is.na(x)))
missing_pct <- round(100 * missing_count / nrow(titanic_raw), 2)
missing_audit <- data.frame(
  Variable = names(missing_count),
  Missing = as.integer(missing_count),
  Percent = as.numeric(missing_pct)
)
print(missing_audit[missing_audit$Missing > 0, ])

# 3. Duplicate and data-type checks
cat("\\nDuplicate rows:", sum(duplicated(titanic_raw)), "\\n")
titanic <- titanic_raw
titanic$Survived <- factor(titanic$Survived, levels = c(0, 1),
                           labels = c("Did not survive", "Survived"))
titanic$Pclass <- factor(titanic$Pclass, levels = c(1, 2, 3),
                         labels = c("First", "Second", "Third"))
titanic$Sex <- factor(titanic$Sex)
titanic$Embarked <- factor(titanic$Embarked,
                           levels = c("C", "Q", "S"),
                           labels = c("Cherbourg", "Queenstown", "Southampton"))

# 4. Missing values
# Age: impute with median within passenger class and sex to preserve broad group differences.
age_group_medians <- aggregate(Age ~ Pclass + Sex, data = titanic_raw,
                               FUN = function(x) median(x, na.rm = TRUE))
for (i in seq_len(nrow(titanic))) {
  if (is.na(titanic$Age[i])) {
    grp <- age_group_medians[
      age_group_medians$Pclass == titanic_raw$Pclass[i] &
      age_group_medians$Sex == titanic_raw$Sex[i], "Age"
    ]
    titanic$Age[i] <- if (length(grp) && !is.na(grp[1])) grp[1] else median(titanic_raw$Age, na.rm = TRUE)
  }
}
# Embarked: mode imputation for the two missing values.
mode_value <- names(which.max(table(titanic_raw$Embarked)))
mode_label <- c(C = "Cherbourg", Q = "Queenstown", S = "Southampton")[mode_value]
titanic$Embarked[is.na(titanic$Embarked)] <- mode_label
# Cabin has very high missingness; retain information via a flag, then remove sparse raw field.
titanic$CabinKnown <- ifelse(is.na(titanic_raw$Cabin), 0, 1)
titanic$Cabin <- NULL

# 5. Feature engineering and encoding
titanic$FamilySize <- titanic_raw$SibSp + titanic_raw$Parch + 1
titanic$IsAlone <- ifelse(titanic$FamilySize == 1, 1, 0)
# One-hot encode selected nominal variables (excluding the target from predictors).
encoded <- model.matrix(~ Sex + Embarked + Pclass - 1, data = titanic)
encoded <- as.data.frame(encoded)
# Bind encoded columns to a modelling table; original categorical fields are retained in titanic.
model_ready <- cbind(
  titanic[, c("Survived", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "CabinKnown")],
  encoded
)

# 6. Outlier detection using the IQR rule (investigate, do not automatically delete)
q1 <- quantile(titanic$Fare, 0.25, na.rm = TRUE)
q3 <- quantile(titanic$Fare, 0.75, na.rm = TRUE)
iqr_fare <- IQR(titanic$Fare, na.rm = TRUE)
lower_fence <- q1 - 1.5 * iqr_fare
upper_fence <- q3 + 1.5 * iqr_fare
fare_outliers <- titanic[titanic$Fare < lower_fence | titanic$Fare > upper_fence, ]
cat("\\nFare IQR fences:", lower_fence, "to", upper_fence, "\\n")
cat("Fare observations flagged by IQR:", nrow(fare_outliers), "\\n")
# Fare outliers are retained because high fares can be legitimate first-class tickets.

# 7. Normalization (min-max) for numeric predictor examples
minmax <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (diff(rng) == 0) return(rep(0, length(x)))
  (x - rng[1]) / (rng[2] - rng[1])
}
model_ready$Age_MinMax <- minmax(model_ready$Age)
model_ready$Fare_MinMax <- minmax(model_ready$Fare)
model_ready$FamilySize_MinMax <- minmax(model_ready$FamilySize)

# 8. Preliminary analysis
cat("\\nSurvival counts:\\n")
print(table(titanic$Survived))
cat("\\nSurvival rate by sex:\\n")
print(round(prop.table(table(titanic$Sex, titanic$Survived), 1) * 100, 1))
cat("\\nSurvival rate by passenger class:\\n")
print(round(prop.table(table(titanic$Pclass, titanic$Survived), 1) * 100, 1))
cat("\\nCorrelation matrix for numeric variables:\\n")
num_data <- titanic_raw[, c("Survived", "Age", "SibSp", "Parch", "Fare")]
print(round(cor(num_data, use = "pairwise.complete.obs"), 2))
cat("\\nMissing values after cleaning:\\n")
print(sum(is.na(titanic)))
cat("\\nCleaned dimensions:", nrow(titanic), "rows x", ncol(titanic), "columns\\n")

# 9. Visualizations (saved as PNG files in the working directory)
p1 <- ggplot(titanic_raw, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white", na.rm = TRUE) +
  labs(title = "Passenger age distribution", x = "Age (years)", y = "Passenger count") +
  theme_minimal()
ggsave("age_distribution.png", p1, width = 7, height = 4, dpi = 150)

p2 <- ggplot(titanic, aes(x = Sex, fill = Survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(title = "Survival proportion by sex", x = "Sex", y = "Proportion", fill = "Outcome") +
  theme_minimal()
ggsave("survival_by_sex.png", p2, width = 7, height = 4, dpi = 150)

p3 <- ggplot(titanic, aes(x = Pclass, fill = Survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = function(x) paste0(round(x * 100), "%")) +
  labs(title = "Survival proportion by passenger class", x = "Passenger class",
       y = "Proportion", fill = "Outcome") +
  theme_minimal()
ggsave("survival_by_class.png", p3, width = 7, height = 4, dpi = 150)

write.csv(titanic, "titanic_cleaned.csv", row.names = FALSE)
write.csv(model_ready, "titanic_model_ready.csv", row.names = FALSE)
cat("\\nSaved titanic_cleaned.csv and titanic_model_ready.csv plus three PNG charts.\\n")
