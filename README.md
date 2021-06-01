# pltr-credit-fraud
Repository for ECMA 31330 final project.

Dumitrescu, et al. (2021) present a new method, named penalized logistic tree regression (PLTR) for discrimative modelling of data with threshold effects. PLTR aims to combine the accuracy of random forests with the ease of interpretability in logistic regression. Dumitrescu, et al. (2021) illustrated the competitive accuracy and interpretability of PLTR in credit scoring, a data with well-known threshold effects. This work aims to demonstrate the robustness of PLTR for modelling threshold effects by implementing PLTR for a different problem, fraud detection. Monte Carlo simulation of credit card transaction data provided evidence for the existence of threshold effects, and the potential for improved accuracy and reliable interpretability with PLTR. The simulated results were backed by experiments on real data, with PLTR outperforming random forest and logistic regression, and identifying several bivariate threshold effects with significant marginal effects on fraud detection.

# How to run this code

If the goal is to recreate the entire experiment, first run "Data simulation code by Le Borgne and Bontempi (2021)" and "Feature transformation by Le Borgne and Bontempi (2021)" in pltr_data_preparation.ipynb. This will generate the simulated data for Monte Carlo.

In the event that any imported packages are missing, simply run every import cell in the notebook. They are all individual cells that can be executed, independent of section.

To prep the simulated data for PLTR, run the "Monte Carlo PLTR data preparation" section in pltr_data_preparation.ipynb.

To run the PLTR algorithm, run mc_pltr.R.

To analyze the coefficients, run "PLTR Coefficient Analysis" section in pltr_data_preparation.ipynb. You only need to run the first cell.

For the real-world data, run "Calulating threshold effects" section in pltr_data_preparation.ipynb to add threshold effects to the real-world data. After it has been generated as "creditcard_thresh.csv," run pltr.R to calculate the PLTR.

To analyze the coefficients, run "PLTR Coefficient Analysis" section in pltr_data_preparation.ipynb. Skip the first cell.

To calculate the logistic regression and random forests, run the Python scripts in /Projects/. The files are named appropriately.
