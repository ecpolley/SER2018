## simulation for incorrect cross-validation, where the pre-filtering of variables is outside the cross-validation.

# data generating function
sim_data <- function(N) {
	Y <- rbinom(N, 1, .4)
	W <- matrix(rnorm(N*1000, 0, 1), ncol = 1000, nrow = N)
	OUT <- data.frame(Y, W)
}

library(SuperLearner) # not required here, but provides structure for cross-validation
library(ModelMetrics) # functions for estimating loss functions


## example single case
set.seed(42)
dat <- sim_data(N = 300)

# select variables with univariate association with outcome from entire dataset
filtered <- screen.corP(Y = dat$Y, X = dat[, -1], family = binomial(), obsWeights = 1, id = 1, method = 'pearson', minPvalue = 0.1)
dat_filt <- data.frame(Y = dat$Y, dat[, -1][, filtered])

# estimate 10-fold cross-validated risk using randomForest algorithm using only filtered variables (incorrect)
fit_sl_filt <- SuperLearner(Y = dat_filt$Y, X = dat_filt[, -1], family = binomial(), SL.library = "SL.randomForest", cvControl = list(V=10), method = "method.NNloglik")

# estimate 10-fold cross-validated risk with variable filtering nested within the cross-validation splits (correct)
sl.library <- list(c("SL.randomForest", "screen.corP"))
fit_sl <- SuperLearner(Y = dat$Y, X = dat[, -1], family = binomial(), SL.library = sl.library, cvControl = list(V=10), method = "method.NNloglik")

# incorrect
auc(actual = dat$Y, predicted = fit_sl_filt$Z[, 1]) # AUC
ce(actual = dat$Y, predicted = as.numeric(fit_sl_filt$Z[, 1]>0.5))  # classification error

# correct
auc(actual = dat$Y, predicted = fit_sl$Z[, 1]) # AUC
ce(actual = dat$Y, predicted = as.numeric(fit_sl$Z[, 1]>0.5))  # classification error

## resub
auc(actual = dat$Y, predicted = fit_sl$SL.predict[, 1]) # AUC
ce(actual = dat$Y, predicted = as.numeric(fit_sl$SL.predict[, 1]>0.5))  # classification error



## now run the simulation
set.seed(20180619)
nSim <- 100 # number of simulation replicates
OUT <- matrix(NA, nrow = nSim, ncol = 6)  # matrix to save results, each row is a simulation
colnames(OUT) <- c("AUC_filtered", "CE_filtered", "AUC_correct", "CE_correct", "AUC_resub", "CE_resub")

for(ii in seq(nSim)) {
	# generate dataset
	dat <- sim_data(N = 300)
	# pre-filtering step, inccorrect outside of CV folds
	filtered <- screen.corP(Y = dat$Y, X = dat[, -1], family = binomial(), obsWeights = 1, id = 1, method = 'pearson', minPvalue = 0.1)
	dat_filt <- data.frame(Y = dat$Y, dat[, -1][, filtered])

	fit_sl_filt <- SuperLearner(Y = dat_filt$Y, X = dat_filt[, -1], family = binomial(), SL.library = c("SL.randomForest", "SL.mean"), cvControl = list(V=10), method = "method.NNloglik")

	sl.library <- list(c("SL.randomForest", "screen.corP"), "SL.mean")
	fit_sl <- SuperLearner(Y = dat$Y, X = dat[, -1], family = binomial(), SL.library = sl.library, cvControl = list(V=10), method = "method.NNloglik")

	# incorrect
	OUT[ii, 1] <- auc(actual = dat$Y, predicted = fit_sl_filt$Z[, 1])
	OUT[ii, 2] <- ce(actual = dat$Y, predicted = as.numeric(fit_sl_filt$Z[, 1]>0.5))  

	# correct
	OUT[ii, 3] <- auc(actual = dat$Y, predicted = fit_sl$Z[, 1])
	OUT[ii, 4] <- ce(actual = dat$Y, predicted = as.numeric(fit_sl$Z[, 1]>0.5)) 

	## resub
	OUT[ii, 5] <- auc(actual = dat$Y, predicted = fit_sl$library.predict[, 1]) 
	OUT[ii, 6] <- ce(actual = dat$Y, predicted = as.numeric(fit_sl$library.predict[, 1]>0.5))  
}

summary(OUT)


# plot the results
library(ggplot2)

df <- data.frame(Method = rep(c("Correct CV", "Incorrect CV"), each = nSim), AUC = c(OUT[, 3], OUT[, 1]))
df_mean <- data.frame(Method = c("Correct CV", "Incorrect CV"), AUC = c(mean(OUT[, 3]), mean(OUT[, 1])))

g <- ggplot(df, aes(AUC, fill = Method, group = Method)) + geom_histogram() + geom_vline(data = df_mean, aes(xintercept = AUC), linetype = "dashed") + theme_bw() + ggtitle("Comparison of Cross-Validated AUC\nTrue Value at 0.5") + xlim(0, 1) + ylab("") + xlab("10-Fold Cross-Validated AUC")