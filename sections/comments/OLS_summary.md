| Variable         | OLS 1 | OLS 2 | OLS 3 | OLS 4 |
|------------------|------------------|------------------------|------------------------|------------------------|
| Intercept        | -0.7976          | N/A                    | N/A                    | N/A                    |
| treatUberX       | 0.0382           | -0.0354                | 0.0075                 | -0.0309                |
| popestimate      | -0.9271          | 0.2789                 | -9.98e-08              | -1.038e-07             |
| employment       | 0.9909           | 0.2677                 | 1.769e-07              | 1.737e-07              |
| aveFareTotal     | -0.1277          | -0.0996                | -0.0012                | -0.0012                |
| VRHTotal         | 1.3417           | 0.3052                 | 8.564e-07              | 8.576e-07              |
| VOMSTotal        | -0.2376          | 0.2314                 | 0.0005                 | 0.0005                 |
| VRMTotal         | 0.0688           | 0.2664                 | -2.596e-08             | -2.602e-08             |
| gasPrice         | 0.2136           | -0.0407                | -0.0096                | -0.0101                |


Overall, the OLS models indicate that we cannot definitively determine whether the effect of Uber on public transit was complementary or supplementary. The results are not robust across regressions, and therefore, we cannot establish causality. This inconsistency suggests that further analysis with more rigorous methods is needed to understand the true impact of Uber on public transit ridership. Below, we try to address these problems using LASSO and double LASSO regression techniques.