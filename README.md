# Optimization

## Package dependencies:
- torch
- pandas
- matplotlib
- seaborn
- tqdm
-scikit-learn

## Before you start 
Make sure to install the above listed libraries 
All scripts must be run from the home folder

## Figures 1 and 2
Optimize a simple time-varying function: $$\log (a \exp(-b_t x + b_t x))$$
```bash simple_function_experiments/run_all.sh```

```bash figures/make_figure1.sh```

```bash figures/make_figure2.sh```

## Figure 3
Optimize a simple positive definite quadratic function using SAGD and plot the parameters: x, y, z

```bash figures/make_figure3.sh ```

## Figure 4
Online Linear Regression Problem 

```bash linear_regression_experiments/run_all.sh ```

```bash figures/make_figure4.sh ```

