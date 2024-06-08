# ECON 434 Project: Uber vs Public Transit

## Overview

In this project, we explore whether Uber complements (helps) or substitutes (hurts) public transit. This analysis builds on the work by Hall, Palsson, and Price (2018) in their paper "Is Uber a Substitute or Complement for Public Transit?" published in the Journal of Urban Economics. Our goal is to expand on their findings and understand the dynamics between Uber and public transit using a comprehensive dataset that includes various metrics related to public transit agencies and the Metropolitan Statistical Areas (MSAs) where they operate.

## Dataset

The dataset used in this project consists of monthly observations for public transit agencies across different MSAs. The key variables included are:

1. **UPTTotal**: Total number of rides for the public transit agency.
2. **treatUberX**: Dummy variable for Uber presence in the corresponding MSA.
3. **treatGTNotStd**: Google search intensity for Uber in the corresponding MSA.
4. **popestimate**: Population in the corresponding MSA.
5. **employment**: Employment in the corresponding MSA.
6. **aveFareTotal**: Average fare for the public transit agency.
7. **VRHTTotal**: Vehicle hours for the public transit agency.
8. **VOMSTotal**: Number of vehicles employed by the public transit agency.
9. **VRMTotal**: Vehicle miles for the public transit agency.
10. **gasPrice**: Gas price in the corresponding MSA.

## Research Questions

1. Does Uber serve as a substitute or complement to public transit?
2. How does Uber's presence affect public transit ridership over time?
3. Are there differences in Uber's impact on transit agencies based on city size and transit agency characteristics?

## Methodology

We employ various regression models to analyze the effect of Uber on public transit ridership. The primary methods used include:

1. **OLS Regressions**: Basic and extended OLS models to capture the impact of Uber presence and intensity on public transit ridership.
2. **LASSO Regressions**: To handle high-dimensional data and interaction terms.
3. **Double-LASSO Regressions**: To robustly estimate the causal effects of Uber on transit ridership by considering interactions of variables.

### Key Regression Models

1. `log Y_it = α + D_itβ + W'_itγ + ε_it`
2. `log Y_it = η_i + τ_t + D_itβ + W'_itγ + ε_it`
3. `log Y_it = η_i + τ_t + D_itβ1 + D_itP_itβ2 + W'_itγ + ε_it`
4. **Double-LASSO**: Variations to include interaction terms and ensure robust coefficient estimates.

## Findings

To be discussed...

## References

- Hall, J. D., Palsson, C., & Price, J. (2018). Is Uber a substitute or complement for public transit? *Journal of Urban Economics, 108*, 36-50. [Link to paper](https://doi.org/10.1016/j.jue.2018.09.003).

## How to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/ECON446-ORG/project_434.git
    ```
2. Navigate to the project directory:
    ```bash
    cd project_434
    ```

3. Create the Conda environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

4. Activate the Conda environment:
    ```bash
    conda activate myenv
    ```

5. Run the analysis scripts:
    ```bash
    python merged.ipynb
    ```
