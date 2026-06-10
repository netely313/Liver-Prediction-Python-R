# =============================================================================
# Liver Disease Prediction - Logistic Regression (Explanatory Model)
# -----------------------------------------------------------------------------
# Goal: identify which clinical blood markers are associated with liver disease.
# Method: binary logistic regression (glm, binomial family) with model
#         diagnostics (overall fit, odds ratios, multicollinearity, linearity).
# =============================================================================

# -----------------------------------------------------------------------------
# Required packages
# -----------------------------------------------------------------------------
# install.packages(c("car", "Hmisc"))   # run once if not already installed
library(car)     # vif() - multicollinearity check
library(Hmisc)   # rcorr() - correlation matrix with p-values

# -----------------------------------------------------------------------------
# 1. Load data and remove duplicate rows
# -----------------------------------------------------------------------------
LiverData <- read.csv(file.choose(), header = TRUE)
LiverData <- unique(LiverData)   # drop duplicate patients (583 -> 570 rows)

# -----------------------------------------------------------------------------
# 2. Encode the target as a factor
# -----------------------------------------------------------------------------
# The first level is the reference (baseline); the model estimates the
# probability of the second level, i.e. P(Liver Disease).
LiverData$Selector <- factor(LiverData$Selector,
                             levels = c("No Liver Disease", "Liver Disease"))

# -----------------------------------------------------------------------------
# 3. Fit the logistic regression model
# -----------------------------------------------------------------------------
LiverModel.1 <- glm(Selector ~ TB + DB + Alkphos + Sgpt + Sgot + TP + ALB,
                    data = LiverData, family = binomial())

summary(LiverModel.1)

# -----------------------------------------------------------------------------
# 4. Overall model significance (likelihood-ratio / chi-square test)
# -----------------------------------------------------------------------------
# Compare null deviance vs residual deviance to test whether the model as a
# whole explains the outcome better than an intercept-only model.
modelChi <- LiverModel.1$null.deviance - LiverModel.1$deviance
modelChi
chidf <- LiverModel.1$df.null - LiverModel.1$df.residual
chidf
chisq.prob <- 1 - pchisq(modelChi, chidf)   # p-value of the overall model
chisq.prob

# -----------------------------------------------------------------------------
# 5. Odds ratios and 95% confidence intervals
# -----------------------------------------------------------------------------
# Coefficients are on the log-odds scale; exponentiate to get odds ratios.
LiverModel.1$coefficients
exp(LiverModel.1$coefficients)
exp(confint(LiverModel.1))

# -----------------------------------------------------------------------------
# 6. Multicollinearity check (Variance Inflation Factor)
# -----------------------------------------------------------------------------
# VIF > 5-10 signals problematic multicollinearity; 1/VIF is the tolerance.
vif(LiverModel.1)
1 / vif(LiverModel.1)

# -----------------------------------------------------------------------------
# 7. Correlation matrix of numeric features (with p-values)
# -----------------------------------------------------------------------------
num_data <- LiverData[sapply(LiverData, is.numeric)]
result <- rcorr(as.matrix(num_data))
result$r   # correlation coefficients
result$P   # p-values

# -----------------------------------------------------------------------------
# 8. Linearity of the logit (Box-Tidwell approach)
# -----------------------------------------------------------------------------
# Add an interaction term X * log(X) for each continuous predictor. If these
# terms are non-significant, the linearity-of-the-logit assumption holds.
LiverData$logTB      <- log(LiverData$TB)      * LiverData$TB
LiverData$logDB      <- log(LiverData$DB)      * LiverData$DB
LiverData$logAlkphos <- log(LiverData$Alkphos) * LiverData$Alkphos
LiverData$logSgpt    <- log(LiverData$Sgpt)    * LiverData$Sgpt
LiverData$logSgot    <- log(LiverData$Sgot)    * LiverData$Sgot
LiverData$logTP      <- log(LiverData$TP)      * LiverData$TP
LiverData$logALB     <- log(LiverData$ALB)     * LiverData$ALB

LiverTest.1 <- glm(Selector ~ TB + DB + Alkphos + Sgpt + Sgot + TP + ALB
                   + logTB + logDB + logAlkphos + logSgpt
                   + logSgot + logTP + logALB,
                   data = LiverData, family = binomial())

summary(LiverTest.1)

# Joint test of all Box-Tidwell terms: non-significant -> linearity holds.
anova(LiverModel.1, LiverTest.1, test = "Chisq")
