In this project, we study whether Uber complements (helps) or substitutes (hurts) public transit. On the one hand, Uber can substitute public transit if riders decide to choose Uber instead of public transit. On the other hand, Uber can complement public transit if riders take Uber from home to public transit stop, which can make public transit more attractive than driving a car. The net effect is unclear and is a subject of intense policy debate.

We will expand on the original set of results presented in Hall, Palsson, and Price (2018), “Is Uber a substitute or a complement for public transit,” *Journal of Urban Economics*, which is available on the class website. We will use their dataset, which is also available on the class website. In the dataset, a unit of ob- servation is a public transit agency in a given year-month. The dataset includes information on both the transit agencies and on the Metropolitan Statistical Areas (MSA) where they operate. For each time period, the dataset contains values for the following variables:

1. $UPTTotal$ – the number of rides for the public transit agency;
2. $treatUberX$ - a dummy for Uber presence in the corresponding MSA;
3. $treatGTNotStd$ - a variable measuring google search intensity for Uber in the corresponding MSA;
4. $popestimate$ - population in the corresponding MSA;
5. $employment$ - employment in the corresponding MSA;
6. $aveFareTotal$ - average fare for the public transit agency;
7. $VRHTTotal$ - vehicle hours for the public transit agency;
8. $VOMSTotal$ - number of vehicles employed by the public transit agency;
9. $VRMTotal$ - vehicle miles for the public transit agency;
10. $gasPrice$ - gas price in the corresponding MSA.

In this dataset, $treatUberX$ and $treatGTNotStd$ is qualitative and quantitative measures for the same thing: Uber presence in the MSA. We can run regressions using either of these two variables and then check whether results are robust if the other variable is used.

There are two variations in this dataset that allow us to study the effect of Uber on public transit. First, in any given time period, Uber is present in some MSAs but not others. We can thus study the effect of Uber by comparing these MSAs. Second, for any given MSA, we have data on time periods both before and after Uber was introduced in this MSA. We can thus study the effect of Uber by comparing these time periods. By working with panel data, we are able to employ both variations at the same time.

To study the effect of Uber on public transit, we let $Y_{it}$ be $UPTTotal$, $D_{it}$ be either $treatUberX$ or $treatGTNotStd$, and $W_{it}$ be the vector including remaining variables: $popestimate$, $employment$, $aveFareTotal$, $VRHTTotal$, $VOMSTotal$, $VRMTotal$, $gasPrice$. We then run the following regressions:
