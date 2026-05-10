    1  Forecasting Macroeconomic Variables: A Systematic
    2  Comparison of Machine Learning Methods


                          1
    3    Teja Konduri and Qian Li2

    4         1University of Notre Dame
    5    2Indiana Legislative Services Agency

    6               July 17, 2024



 7 Abstract

 8 This paper evaluates the performance of an extensive set of machine learning algo-
 9 rithms in forecasting macroeconomic variables relative to baseline econometric models.
10 We conduct a pseudo-out-of-sample forecast for fifteen real, nominal, and financial vari-
11 ables. The findings can be summarized in three points. First, machine learning models
12 perform better than the benchmark model in forecasting real variables but worse than
13 the baseline models in forecasting nominal variables (price indices) and financial vari-
14 ables. Second, machine learning models forecast better than benchmark models during
15 periods of high volatility, like recessions and the COVID-19 pandemic. Third, models
16 that employ dimension reduction frequently appear in the top five most accurate models
17 in forecasting real variables, especially at longer horizons.

18

19 Keywords: Macroeconomic Indicators Forecast, Autoregressive, Random Walk, Ma-
20 chine Learning Methods, Data-rich Environment, Ensemble Methods, Dimension Reduc-
21 tion, Diffusion Index

        1

22 1 Introduction

23 The evolution of economic prediction methods has shifted from traditional econometric
24 models, like Auto-Regressive (AR) forecasts, to advanced machine learning techniques,
25 marking a significant change in economic forecasting. This shift is driven by the grow-
26 ing complexity and abundance of data, demanding powerful analytical tools capable of
27 capturing nonlinear patterns and efficiently using vast datasets. While traditional econo-
28 metric models have laid the groundwork, they are often hampered by the “curse of di-
29 mensionality,” where the increase in the number of predictors leads to an exponential
30 increase in the parameters that need to be estimated. This proliferation of parameters
31 increases the risk of overfitting, where models may fit the training data very closely but
32 fail to generalize to new, unseen data due to their sensitivity to noisy or unrepresentative
33 training samples. This limitation is particularly acute in forecasting economic indicators
34 such as industrial production, employment rates, and inflation, which are key inputs for
35 policy decisions and economic analysis.
36 Machine learning models offer a promising alternative due to their ability to integrate
37 and learn from vast amounts of data and their flexibility in modeling complex, non-linear
38 relationships often present in macroeconomic environments. Unlike traditional models
39 that require assumptions about the functional form of relationships between variables,
40 machine learning models can adaptively learn these relationships without pre-selection or
41 fixed weighting, potentially allowing for accurate forecasting. For instance, studies like
42 Serrano & Hoesli (2007) and Choudhary & Haider (2012) show that machine learning
43 models can outperform traditional methods like Vector Auto Regressions and autoregres-
44 sive processes by responding dynamically to shifts in economic conditions.
45 Despite the growing use of machine learning in economic analysis, there remains a lack
46 of comprehensive studies that systematically compare the performance of these models
47 across a wide array of macroeconomic variables. Our study aims to fill this gap by utilizing
48 a diverse set of machine learning models to forecast key economic indicators and compare
49 their performance with traditional benchmarks. We build upon the work of Kotchoni et al.
50 (2019) by extending their analysis to incorporate a larger universe of machine learning

        2

51 approaches and a more extensive set of economic indicators, providing new insights into
52 their out-of-sample forecasting performance in a sample that also includes the COVID-19
53 pandemic.
54 The recent literature has increasingly recognized the potential of machine learning
55 techniques in enhancing economic forecasting, presenting a formidable challenge to tra-
56 ditional econometric methods. For instance, Kotchoni et al. (2019) demonstrate the
57 superior accuracy of regularized data-rich models in forecasting crucial macroeconomic
58 variables, highlighting the necessity of diverse modeling approaches in data-rich environ-
59 ments to improve forecast accuracy. Similarly, Milunovich (2020) reveal how machine
60 learning and deep learning algorithms could outperform traditional benchmarks like the
61 random walk model in predicting real estate indices. Furthermore, Goulet Coulombe
62 et al. (2022) detail how machine learning excels in capturing nonlinear dynamics – a key
63 feature that traditional models often overlook.
64 Our study confirms these findings by comparing machine learning models against
65 baseline econometric models. It also expands the analysis by exploring their performance
66 across a larger spectrum of economic indicators, including those impacted by the COVID-
67 19 pandemic. This comprehensive approach allows us to dissect the conditions under
68 which machine learning models excel, particularly in handling nonlinear interactions and
69 large datasets. By doing so, we provide policymakers and economic analysts with deeper
70 insights into the efficacy of various forecasting methods during periods of both stability
71 and significant economic turbulence.
72 We categorize the models considered in our analysis into five distinct groups to provide
73 a structured comparison: traditional benchmark econometric models; individual machine
74 learning models; ensemble machine learning models, which harness the collective power of
75 multiple learners for enhanced accuracy; and two categories through which we reevaluate
76 the machine learning forecasts using dimension reduction techniques.
77 Our results can be summarized as follows: An analysis of the period leading up to
78 the COVID-19 pandemic (1960M1-2019M12) reveals that ML models generally surpass
79 traditional econometric models in forecasting real variables like industrial production and


    3

 80 employment yet fall short when it comes to nominal and financial variables such as CPI
 81 and the S&P 500. Expanding our analysis to include a broader set of variables across real,
 82 nominal, and financial categories, we observe a consistent pattern: ML models tend to
 83 outperform the baseline in forecasting real variables but not nominal and financial ones.
 84 This pattern prompts a deeper exploration into the specific conditions under which ML
 85 models excel. By benchmarking against the Auto Regressive Diffusion Indices (ARDI)
 86 model, we find that the superior performance of ML models in forecasting real variables
 87 cannot be solely attributed to their ability to handle data-rich environments. Instead, as
 88 Goulet Coulombe et al. (2022) highlighted, the inherent nonlinearity within ML models
 89 emerges as a significant factor. Further investigation into periods of high volatility, such
 90 as NBER recessions and a sample that includes the onset of the pandemic in early 2020,
 91 indicates that ML models improve more over the baseline in forecasting real variables
 92 during turbulent periods than normal times. While reducing dimensionality enhances
 93 the forecasting accuracy of certain machine learning models, their performance varies
 94 across variables and horizons. Finally, the machine learning model, Adaptive Boosting
 95 (AdaBoost), and its diffusion index counterpart produce the most stable forecasts over
 96 time for real variables, while the benchmark model is the most stable for forecasting
 97 nominal and financial variables.
 98 The remainder of the chapter proceeds as follows. Section 2 describes the forecasting
 99 framework and targets, the data used in our forecasting exercise, and the empirical eval-
100 uation design, while section 3 presents the forecasting models. In section 4, we discuss
101 the forecasting performance of the key variables. In section 5, we expand our analysis to
102 include new variables to examine whether the pattern across different types of variables
103 uncovered in the previous section holds more broadly and present the forecasting results
104 for these new variables. In section 6, we examine the cause behind the strong perfor-
105 mance of the machine learning models in forecasting real variables. After determining
106 that non-linearities play an important role in machine learning models’ forecasting ability,
107 we examine their performance during NBER recession periods and a sample that includes
108 post-COVID data in section 7. In section 8, we examine the forecasting stability of the


    4

    109 models, and in section 9 we conclude our analysis and discuss future steps for research.


110 2 Forecasting Environment and Evaluation

111 This section introduces our general forecasting framework and defines the forecasting
112 targets. Then, we describe the data and the forecast evaluation methodology.


113 2.1 General Forecasting Framework and Forecasting Targets

114 In this chapter, we consider the following general framework for our predictive models
115 from Mullainathan & Spiess (2017):


              arg min X L(yₜ₊ₕ − f (Xₜ; θ)),    t = 1, . . . , T.                                 (1)
              θ     t
    116        where, yₜ₊ₕ is the target, i.e., the variable we predict h periods into the future and
117 Xₜ is the N - DImensional vector of predictors. We minimize the quadratic loss function,
118 L, by choosing the parameters θ of the function f (.). f (.) models the predictors’ space
119 linearly or non-linearly using a flexible functional form. In this chapter, the optimal
120 forecast is the conditional expectation E(yₜ₊ₕ|Xₜ).
    121            We now describe the targets of our forecasting exercise. Let Yₜ denote an economic
    122  time series we want to predict. Before forecasting Yₜ, we stationarize it by following Stock
    123         & Watson (2002) and McCracken & Ng (2016). In this chapter, we make direct forecasts.
    124   i.e., we forecast a variable’s h−period ahead value directly from the current period. If yₜ
    125    is the stationary transformation of Yₜ, we forecast its annualized average over the period
    126   [t + 1, t + h] given by:        h
              y(h)     = 1200            X y                                                      (2)
              t+h         h              k=1    t+k
    127             We multiply the monthly growth by 12 to make it annualized and multiply by 100 to
    128      calculate the percentage. For more information, see Kotchoni et al. (2019). We deal with
    129   three separate types of series:

    130   1. Specifically, if yₜ ≡ ln Yₜ is stationary, we forecast equation (2).

              5

    131   2. If yₜ ≡ ln Yₜ − ln Yt−1 is stationary, i.e., if ln Yₜ is integrated of order 1 – I (1), then
    132       we forecast:    y(h) = 1200 ln Yᵗ⁺ʰ                                                     (3)
                         t+h     h     Yₜ

    133   3. If yₜ ≡ (ln Yₜ − ln Yt−1) − (ln Yt−1 − ln Yt−2) = ∆² ln Yₜ is stationary, i.e., if ln Yₜ  is
    134       integrated of order 2 – I (2), then we forecast:


              y(h)     = 1200     ln     Yₜ₊ₕ     − ln     Yₜ                                         (4)
              t+h         h             Yt+h−1      Yt−1

    135   2.2 Data

136 In our study, we utilize data from FRED-MD, an expansive monthly macroeconomic
137 database, to evaluate and compare the performance of the forecasting models outlined
138 in the next section. However, the FRED-MD dataset, post-June 2016, does not include
139 the seven Institute for Supply Management (ISM) manufacturing indices. Economists
140 and policymakers recognize these indices, which serve as measures of the Purchasing
141 Managers’ Index (PMI), as a crucial indicator of the health of the US economy (see
142 Kauffman (1999)).
    143   Given the significance of these measures, we sourced these series from the YCharts
    144                  database and incorporated them into the revised FRED-MD panel. The macroeconomic
    145               panel we employ in our study comprises 134 monthly macroeconomic and financial time
    146   series from January 1960 to December 2023.


147 2.3 Forecasting Methodology

148 2.3.1 Pseudo Out-Of-Sample Forecasting Design

149 For our forecasting exercise, we adopt a pseudo-out-of-sample approach for the period
150 from January 1970 to December 2019. Our forecast horizons span 1, 3, 6, 9, and 12
151 months, with 593 − h evaluation periods for each forecasting horizon h. We use rolling
152 windows for model estimation, with a window size of 120 − h months. For example,
153 the forecast for January 1970 is based on data from January 1960 to December 1969;

        6

154 similarly, the forecast for February 1970 relies on data from February 1960 to January
155 1970, and so on. This rolling window approach ensures consistency and eases comparison
156 across different models while dynamically adapting to use the latest available data.
157 To simplify cross-model comparisons, we uniformly use six lags across all evaluation
158 periods and forecasting horizons. Employing six lags, we use information from the recent
159 past, which is useful in forecasting a variable; at the same time, we do not overwhelm
160 the model with too many parameters that come with higher lags. We determine the
161 hyperparameters for the machine learning models, such as number of neighbors, kernel
162 choice, the maximum number of decision splits, and learning rate, based on conventional
163 heuristics.

164 2.3.2 Variables of Interest

165 Our analysis focuses on the four macroeconomic indicators forecasted by Kotchoni et al.
166 (2019). These variables, along with their mnemonics, are industrial production (IND-
167 PRO), employment (PAYEMS), consumer price index (CPIAUCSL), and the S&P 500
168 index (S&P 500). Industrial production and employment are real variables, reflecting
169 actual economic outputs and labor market conditions. In contrast, the CPI represents
170 the price levels in the economy and is a nominal variable. The S&P 500 reflects stock
171 market valuations and hence is a financial variable.
172 We treat the logarithms of industrial production, employment, and S&P 500 as I (1)
173 variables, indicating that their month-on-month growth rates are stationary. Conversely,
174 Stock & Watson (2002), McCracken & Ng (2016), and Kotchoni et al. (2019), classify the
175 logarithm of the CPI as I (2), which implies that the changes in CPI’s growth rate–or the
176 inflation rate’s growth rate–are stationary.

177 2.3.3 Forecast Evaluation Metrics

178 Following standard economic forecasting practices, we evaluate the accuracy of our point
179 forecasts using the Root Mean Square Prediction Error (RMSPE). The RMSPE is cal-
180 culated as:


    7

                 v
                 u
                 u    X
         RMSPE = t 1 N  (ˆ
                  N t=1 yₜ − yₜ)²

                                                                            yₜ denotes the
    181  where N is the number of forecast evaluation points for each horizon h, ˆ
182 forecasted values, and yₜ represents the actual observed values at time t. The RMSPE
183 calculates the square root of the average of the squared differences between the forecasted
184 and actual values, providing a measure of the prediction accuracy of a model. The
185 RMSPE penalizes large forecast errors.
    186  Additionally, we employ the Diebold & Mariano (1995) test – the DM test – to sta-
187 tistically compare the predictive accuracy of our models against the baseline econometric
188 model. This test assesses whether the difference in forecasting errors between two models
189 is statistically significant, providing a robust method to ascertain if one model consis-
190 tently outperforms another across our forecasting horizons. In the subsequent section,
191 we describe the different models we use to forecast the macroeconomic indicators, focusing
192 on their distinguishing features.


193 3 Model Universe

194 We employ 24 time series and machine learning models to forecast macroeconomic in-
195 dicators, organizing them into five distinct categories (see Table 1): baseline models,
196 individual machine learning models, ensemble machine learning models, individual ma-
197 chine learning models using dimension reduction, and ensemble machine learning models
198 using dimension reduction.


199 3.1 Baseline Models

200 We use the autoregressive direct (ARD) model as our benchmark (baseline) model fol-
201 lowing Stock & Watson (2002) and Kotchoni et al. (2019). ARD model is a univariate
202 forecasting method that predicts a variable’s h−period forecast using its current and
203 lagged values. The model is mathematically expressed as follows:


         8

           X
           L
          y(h) = α(h) + ρ(h)y + e
          t+h        l  t−l+1     t+h,     t = 1, . . . , T,       (5)
          l=1
    204   for h ≥ 1 and L ≥ 1. We use this model as our benchmark because of its simplicity. We
    205   standardize our analysis by setting L = 6 for all models.
    206   Furthermore, we benchmark financial variables against the Random Walk (RW) model
    207   without drift, a convention in finance literature.


208 3.2 Individual Machine Learning Models

209 Individual machine learning models are the first of our four groups of machine learning
210 models. This category includes models like k-Nearest Neighbors(k-NN), decision trees,
211 and Support Vector Regressions (SVR) with diverse kernels.
212 K-Nearest Neighbors (kNN): This nonparametric method does not explicitly as-
213 sume a specific form for the function f (xₜ). Instead, the forecasted outcome y (xₜ) is
214 derived as the weighted average of the targets of the k nearest data points to xₜ. The
215 optimal value of k depends on the bias-variance trade-off, with a common heuristic being
        √
216 k = ⌊ N ⌋, where N is the size of the training dataset and ⌊x⌋ is the greatest integer ≤ x.
217 In our analysis, with a training dataset spanning ten years or 120 months, we set k = 10
218 to optimize performance. For distance measurement, we employ the Euclidean metric
219 and introduce two weighting schemes for the target forecast: “kNN (uniform),” where
220 all neighbors are equally weighted, and “kNN (inverse),” where weights are inversely re-
221 lated to their distance, emphasizing nearer neighbors more significantly. By assessing the
222 “neighborhood” of a given data point, kNN captures these spatial dependencies, which
223 are not explicitly modeled in traditional parametric approaches. This enables the algo-
224 rithm to adaptively respond to the data’s intrinsic structure, making it especially effective
225 when economic variables show significant spatial continuity or clustering.
226 Decision tree regression: This model utilizes a tree-structured approach to fore-
227 cast future data, adept at identifying nonlinear relationships and interactions between
228 variables without predetermined functional forms. Originating from Quinlan (1986)’s
229 ID3 algorithm, decision trees use a top-down, greedy search to construct decision rules

        9

230 directly from data, optimizing for continuous outputs with strategies like Standard Devi-
231 ation Reduction. The choice of hyperparameters – limiting to 20 leaf nodes and requiring
232 a minimum of 3 samples per leaf – strikes a balance between simplicity and depth, ensur-
233 ing interpretability, computational efficiency, and generalizability. Using hyperparameter
234 tuning methods such as grid search, randomized search, or Bayesian optimization fa-
235 cilitates discovering an optimal model configuration that is accurate and resistant to
236 overfitting.
237 Support Vector Regression (SVR): SVR is a machine learning technique designed
238 specifically for regression analysis. Unlike traditional linear regression, which aims to fit
        P
239 a line through data points, SVR aims to find a function f (x) = ni=1(αi − αi∗)K(xi, x) + b
240 that approximates the relationship between input features x and target values y within
241 a certain margin of error ϵ. This function f (x) is created from the training data using
242 kernel functions K(x, x′), which enable the model to capture nonlinear relationships by
243 mapping inputs into a higher - DImensional space. We use the following four kernels:

244 • Linear Kernel: K(x, x′) = ⟨x, x′⟩

245 • Polynomial Kernel: K(x, x′) = (γ⟨x, x′⟩+r)ᵈ where γ, r, and d are kernel parameters
246 that control the shape of the polynomial.

247 • Radial Basis Function (RBF) Kernel: K(x, x′) = exp(−γ∥x − x′∥²), , with γ influ-
248 encing the spread of the RBF kernel.

249 • Sigmoid Kernel: K(x, x′) = tanh(γ⟨x, x′⟩ + r)

250 The selection of the kernel function K(xx′) and the tuning parameters, like γ, d, and
251 r, play a role in determining how well the model fits the data. SVR optimizes a set of
252 coefficients αi and αi∗ along with a bias term b to ensure the model is as flat as possible
253 while fitting within an epsilon tube around the training data. This approach allows SVR
254 to effectively handle high - DImensional datasets where traditional regression models face
255 challenges.
256 However, SVR has limitations in terms of efficiency. Training SVR models with non-
257 linear kernels can be computationally intensive. The performance of SVR algorithms

        10

258 is sensitive to the scale of input features. Thus, pre-processing steps such as scaling
259 become essential. Standardizing predictors and outcomes helps prevent any variable
260 from influencing the model due to its magnitude. After estimation, we reverse the scaling
261 process to facilitate model comparison using the Root Mean Squared Prediction Error
262 (RMSPE) metric.


263 3.3 Ensemble Machine Learning Models

264 Ensemble methods enhance prediction accuracy and robustness by combining the outputs
265 of multiple base estimators. This approach leverages the strength of various algorithms
266 to offset a single model’s weakness. The most popular ensemble methods for regression
267 include Random Forest, Extreme Gradient Boosting (XGBoost), Adaptive Boosting (Ad-
268 aBoost), and Gradient Boosting, all of which have distinct advantages and implications
269 for the data being forecasted (Dietterich (2000)).
270 Random Forest: Random Forest works by creating decision trees during training
271 and reporting the average prediction of these trees (Breiman (2001)). This approach is
272 highly effective for datasets with complex interactions and nonlinear relationships as it
273 doesn’t rely on the underlying distribution of the data. The key strength of Random
274 Forest lies in its ability to combat overfitting through bagging, a method that combines
275 the results of multiple models to enhance performance (see Breiman (1996)). If Random
276 Forest proves to be the most accurate model, it indicates that the dataset benefits from
277 a model that can handle high - DImensional data and complex interactions between
278 variables.
279 Gradient Boost: Gradient Boosting builds a model iteratively by minimizing loss
280 through gradient descent (Friedman (2001)). This technique offers a way to develop
281 models that evolve gradually over time by focusing on correcting errors from previous
282 iterations. Gradient Boosting is particularly effective for datasets exhibiting complex
283 interaction patterns where gradual improvements are essential.
284 Extreme Gradient Boosting (XGBoost): XGBoost is an advanced version of
285 gradient boosting known for its efficiency, flexibility, and portability (Chen & Guestrin

        11

286 (2016)). It enhances gradient boosting by improving speed and performance while effec-
287 tively handling sparse data. This approach is particularly suitable for situations where
288 both speed and accuracy are crucial, excelling in cases where precision is a top prior-
289 ity. The triumph of XGBoost in a horse race underscores the dataset’s receptiveness to
290 a model that emphasizes enhancements based on errors, underscoring its sensitivity to
291 fine-tuning and regularization to prevent overfitting.
292 AdaBoost (Adaptive Boosting): AdaBoost is a boosting technique aimed at
293 transforming many weak learners into one strong learner (Freund & Schapire (1997)).
294 It adjusts the weights of misclassified instances to ensure subsequent classifiers pay more
295 attention to them. This method works well for imbalanced datasets or those requiring
296 resilience against noise and outliers. If AdaBoost surpasses all other models in perfor-
297 mance, it suggests that the data benefits from iterative instance reweighting, indicating
298 varying degrees of complexity within the data and necessitating adaptive adjustments.


299 3.4 Machine Learning Models Using Dimension Reduction

300 Our dataset comprises over 100 macroeconomic variables. Due to the intricate inter-
301 dependencies between the predictors and the forecasted variables, there is a high risk
302 of overfitting. We integrate diffusion indices (DIs) derived from principal component
303 analysis (PCA) into our forecasting models to address these challenges.
304 Ma & Zhu (2013) and Kotchoni et al. (2019) identify three key methods to improve
305 out-of-sample forecasting accuracy while mitigating overfitting: sparse modeling, regu-
306 larization, and dense modeling. We adopt dense modeling via PCA, which assumes that
307 a few principal components can significantly capture the variance in the data. These
308 components, our DIs, condense the dataset’s vast information into a manageable form,
309 enhancing model efficiency 1. DIs retain the information that has the most predictive
310 power and discard the noise and less informative variability that contributes to overfitting.
311 In categories four and five of table 1, we re-evaluate the forecasts of all the machine
312 learning models using DIs. We select the number of factors for each variable and hori-
       1Stock & Watson (2002)


    12

313 zon based on the panel criteria proposed by Bai & Ng (2002). From the recommended
314 number of factors, we pick the smallest number of factors for parsimony. We expect the
315 revised models, identified with a “DI” suffix, to deliver improved forecasting accuracy by
316 efficiently leveraging the condensed yet informative representation of the data.


317 4 Forecast Results for Key Variables

318 In this section, we present the results for the forecasting accuracy for industrial produc-
319 tion, employment growth, CPI inflation, and S&P 500 index returns, presented in Tables
320 2 to 5. Our analysis spans the entire out-of-sample period from January 1960 to Decem-
321 ber 2019. Each table’s left panel displays the full out-of-sample forecasting results, while
322 the right panel focuses on performance during NBER recessions (i.e., target observation
323 belongs to a recession episode) which we discuss in section 7.1. The baseline model’s RM-
324 SPE occupies the first row of each panel, with subsequent rows comparing the relative
325 RMSPE of machine learning models to this baseline. The relative RMSPE of a model is
326 the ratio of its RMSPE to the RMSPE of the baseline. We underline the best model in
327 terms of relative RMSPE (i.e., the minimum relative RMSPE) for each horizon, and the
328 significance levels for the DM tests are displayed using the conventional notation with
329 three, two, and one asterisks.
330 The forecasting performance for industrial production growth reveals that multiple
331 machine learning models outperform the baseline ARD model at each horizon. Of these
332 models, SVR (RBF) and AdaBoost stand out for their accuracy, aligning with the forecast
333 accuracy of Kotchoni et al. (2019)’s best models. Seven models beat the baseline in
334 the short term at h = 1, with AdaBoost demonstrating superior short-term forecasting
335 abilities than the other models. While for longer horizons at h = 9, 12, we still find 8, 6
336 models beating the baseline. However, the DI-enhanced models, notably kNN variants,
337 perform best suggesting a spatial proximity among industrial production data points and
338 their lags.
339 For employment growth, as shown in Table 3, eight models outperform the baseline


    13

340 in the short term, while only four models outperform the baseline in the long term.
341 Random Forests and AdaBoost initially outperform the baseline, incorporating diffusion
342 indices generally enhancing model performance across various horizons. Medium-term
343 forecasts see SVR (linear) - DI and AdaBoost - DI as front runners. At more extended
344 horizons, kNN (uniform) - DI shows improved RMSPE over the baseline. However,
345 these improvements are not always statistically significant, indicating that while forecast
346 accuracy improves, it does not uniformly exceed baseline performance throughout the
347 period.
348 As detailed in Table 4, all machine learning models fall short of surpassing the baseline
349 ARD model in forecasting CPI inflation accuracy, underscoring the challenges machine
350 learning models face with nominal variables.
351 The forecasting results of S&P 500 returns are outlined in Table 5. While, in principle,
352 we need real-time data vintages for forecasting financial variables, our study relies on
353 the latest available information at the time of forecasting. Like CPI inflation, machine
354 learning models do not outperform the baseline random walk model, with only kNN
355 (uniform) and AdaBoost yielding better predictions than the baseline at select horizons.
356 The RW baseline ranks first at the longer horizons, while it comes fourth, second, and
357 third at h = 1, 3, 6 respectively. Notably, there is no statistically significant difference
358 between the baseline and machine learning forecasts.
359 Overall, our initial forecast results show the superiority of machine learning models in
360 predicting real variables like industrial production and employment. However, machine
361 learning models appear less promising for forecasting nominal and financial variables such
362 as CPI inflation and S&P 500 returns. While PCA, as a dimension reduction technique,
363 aids in forecasting employment, it does not significantly enhance the prediction of other
364 variables.










    14

365 5 Do Forecasting Patterns Hold Across a Larger Set

366 of Variables?

367 5.1 Expanding the Variable Set

368 In the previous section, we saw that machine learning models are more accurate than
369 univariate time series models in forecasting Industrial Production (INDPRO) and Em-
370 ployment (PAYEMS), which are real variables, i.e., quantities. On the other hand, the
371 baseline econometric models are more accurate than the ML models in forecasting CPI,
372 a price index, and S&P 500, a financial index. To investigate whether there is a pattern
373 that machine learning models can outperform baseline models in predicting real variables
374 and under-perform baseline models in predicting nominal and financial variables, we ex-
375 pand our set of variables to include five variables in each category to investigate if the
376 pattern holds.
377 In addition to industrial production and employment, our expanded set of real vari-
378 ables includes real personal income (RPI), the unemployment rate (UNRATE), and real
379 personal consumption expenditure (Real PCE). The industrial sector, together with con-
380 struction, accounts for the bulk of the variation in national output over the course of the
381 business cycle. On the other hand, the three new variables reflect consumer sentiment
382 in the economy. Since consumption contributes to between 60-70% of the GDP, the five
383 real variables we forecast are strong indicators of the economy’s health.
384 We focus on various consumer and producer price indices when expanding our selection
385 of nominal variables. While CPI for all items offers a broad measure of inflation, we
386 also incorporate a less volatile measure of the CPI by excluding volatile food prices.
387 Additionally, we include the Personal Consumption Expenditures Price Index (PCEPI)
388 to complement these measures. Unlike the CPI, which directly assesses consumer out-of-
389 pocket expenses, the PCEPI offers a wider lens on inflation by capturing all goods and
390 services consumed by households, including those paid on their behalf, like healthcare
391 benefits. Finally, the two producer price indices (PPIs) – the PPI for finished consumer
392 goods, and the PPI for crude metals – offer insights into sector-specific inflation pressures

        15

393 that broader indices might not capture. We focus on forecasting the second difference
394 (I(2)) of the logarithm of each price index.
395 Lastly, our expanded selection of financial variables spans an array of indicators such
396 as treasury rates (1 and 10 years) and exchange rates (US/UK and CA/US foreign ex-
397 change rates). The treasury rates provide a spectrum of risk and time preferences in the
398 financial markets, while the foreign exchange rates are critical for assessing international
399 trade dynamics and financial flows.


400 5.2 Forecast Results

401 This section presents the forecasting results for the additional variables in tables 6 – 8.
402 For each variable, we show the baseline RMSPE as well as the relative RMSPEs of the
403 five best models for that variable. We present detailed results in the appendix.
404 Table 6 displays the forecast results for the newly added real variables: real personal
405 income, unemployment rate, and real personal consumption expenditure. We find that
406 the pattern of machine learning models outperforming the baseline continues to hold for
407 all three real variables.
408 Our analysis of the income forecasts reveals that 12 models beat the baseline at h = 1,
409 while seven models are more accurate than the baseline at h = 12. In particular, the SVR
410 (RBF) and AdaBoost models excel in the short to medium term (1, 3, and 6 months).
411 For longer forecast horizons (9 and 12 months), kNN methods incorporating diffusion
412 indices emerge as the most accurate.
413 We next examine the forecasting accuracy of the unemployment rate. At h = 1, eight
414 machine learning models including Random Forests, AdaBoost, and the kNN models us-
415 ing diffusion indices consistently outperform the baseline with a relative RMSPE between
416 0.93 and 0.96. At longer horizons, only the two kNN - DI models consistently outshine
417 the baseline, with their accuracy improving with the horizon from a relative RMSPE
418 decreasing from 0.87 at h = 3 to 0.75 at h = 12. 12 models perform better than the
419 baseline at h = 12, but only two predict significantly differently than the benchmark.
420 For Real PCE, only four ML models outperform the baseline at h = 1, and three

        16

421 perform better at h = 12. In the short run (1 and 3 months), the SVR (RBF) model
422 shows remarkable accuracy, whereas, for medium-term forecasts (6 and 9 months), the
423 kNN (uniform) with diffusion indices takes the lead.
424 The data underscores a consistent trend: machine learning models, particularly those
425 incorporating advanced techniques like diffusion indices, significantly outperform the
426 baseline ARD model across the board for real variables. This finding aligns with our
427 initial hypothesis, affirming the superior predictive power of machine learning models in
428 this context.
429 Table 7 illustrates the forecasting accuracies of the different ML models for the nomi-
430 nal variables – CPI, CPI less food, PCEPI, PPI for finished consumer goods, and PPI for
431 crude metals. Notably, the baseline model consistently outperforms all machine learning
432 models across all forecast horizons, underscoring the ARD model’s robustness in predict-
433 ing these variables.
434 Finally, we present the results for the financial variables in table 8. Our analysis
435 reveals that the RW baseline consistently outperforms all machine learning models in full
436 out-of-sample forecasts. For 1-year Treasury rate (GS1), 3 and 2 models forecast better
437 than the RW baseline at h = 1, 3 respectively but their performance is not significantly
438 different from the baseline. For both S&P 500 and GS1, AdaBoost shows marginal
439 improvement over the random walk baseline at h = 1.
440 These findings suggest that while the RW model remains a strong predictor for fi-
441 nancial variables overall, the AdaBoost model shows promise in specific contexts and
442 horizons.
443 This section confirmed our hypothesis that machine learning models are more accurate
444 than the baseline AR Direct model in forecasting real variables. On the other hand, the
445 baseline is the best in forecasting nominal and financial variables.










    17

446 6 Data-Richness vs Non-Linearities

447 In the previous section, we saw that the machine learning models outperform the baseline
448 when predicting real variables. Now, we will investigate what leads to the better perfor-
449 mance of the ML models over the baseline model. Recall that our baseline model, the
450 Auto Regressive Direct forecast, is a linear univariate model. i.e., the baseline uses only
451 the lags of the predicted variable as the predictors, and the predicted variable is a linear
452 combination of the predictors. Our ML models, on the other hand, are in a data-rich
453 space where they use the 134 variables in the FRED-MD database along with their lags.
454 At the same time, all our ML models take advantage of non-linear relationships between
455 the predictors and the predicted variable to improve forecast accuracy. In this section,
456 we try to find why our ML models are good at forecasting the real variables by examining
457 two dimensions: data-richness and non-linearities.


    458   6.1 A New Baseline

459 To accomplish our goal, we use a new baseline model, the Auto Regressive Diffusion
460 Indices (ARDI), which was first introduced by Stock & Watson (2002). In this model,
461 the diffusion indices are extracted from a set of predictors and then augmented in a direct
462 autoregression. This model is written as:


        y(h) = α(h) + Xᵖʰʸ ρ(h)y + Xᵖʰᶠ β(h)F + e , t = 1, . . . , T
        t+h l t−l+1 l t−l+1 t+h
        l=1 l=1

463 where Fₜ are K(h) consecutive static factors and the superscript h stands for the value
464 of K when forecasting h periods ahead. We use various information criteria based on
465 penalized likelihood, such as AIC and BIC, to determine the number of factors to be
466 included in the predictive regression for each target variable and forecasting horizon. We
467 use the smallest value these criteria return as the optimal value of K(h). To maintain
468 uniformity in the lag selection across all our models, we use six lags in the ARDI model
469 by setting pʰy = pʰf = 6. With this, the h−step ahead forecast is obtained as:



          18

                    y          Xᵖʰʸ           Xᵖʰᶠ
                    ˆ(h) = ˆ
                        α(h) + ρˆ(h)y  +          β
                    T +h|T         l   t−l+1      ˆₗ(h)Ft−l+1
                               l=1            l=1

    470  6.2 Results

471 Table 9 displays the results of the top machine learning models compared with ARDI.
472 Even though the ARDI is a data-rich model, our machine learning models continue to
473 beat it for all five of our real variables. For industrial production, we see that AdaBoost’s
474 RMSPE is only 0.87 times the RMSPE of ARDI for h = 1. For longer horizons, the
475 kNN models have relative RMSPEs less than 1, ranging between 0.83 and 0.72. We also
476 observe that more ML models beat the ARDI baseline than the ARD. At h = 1, 12
477 models outperform the ARDI while only 7 outperformed the ARD. Similarly, at longer
478 horizons of h = 9, 12, we find that 13 and 20 models perform better than the ARDI while
479 only 8 and 6 outperformed the ARD.
    480            For employment, 13 models, including Random Forests, XGBoost, AdaBoost, and
481 their DI counterparts, outperform the ARDI baseline at h = 1, with random forests
482 having the smallest relative RMSPE of 0.83 compared to ARDI. As the horizons get
483 larger, the number of models outperforming the baseline remains at 13, although the
484 best models change to the SVR (linear) - DI and kNN (uniform) - DI.
    485   A similar pattern repeats for real personal income, unemployment rate, and real PCE.
486 20 ML models outperform real personal income’s baseline forecast at h = 1. AdaBoost
487 is the most accurate forecaster for income and unemployment rate, while for real PCE it
488 is the SVR (rbf). As the horizons increase, kNN uniform DI becomes the best predictor
489 for all three variables. This indicates that the kNN uniform DI excels at capturing
490 spatial patterns or dependencies in the data, maintaining temporal stability, handling
491 data complexity, incorporating relevant features, and exhibiting flexibility in modeling
492 the data dynamics over time. At h = 12, 21, 13, and 22 models outperform the ARDI
493 baseline for the three variables respectively.
    494  Overall, in this section, we observe four things. First, at least one machine learning
    495  (ML) model surpasses the ARDI baseline for all variables at each forecasting horizon.

                                 19

496 This suggests that, despite the richness of the data, ML models are adept at leverag-
497 ing either spatial dependencies or non-linearities in forecasting real variables. Second,
498 at h = 1, multiple ML models can predict the value of these variables better than the
499 ARDI. This indicates that the ARDI model might not adequately capture the short-term
500 dynamics or may be too rigid in its assumptions about the data. Other models, includ-
501 ing kNN uniform DI, might be more flexible or better suited to capture the short-term
502 fluctuations, potentially by incorporating spatial dependencies or leveraging non-linear
503 relationships in the data. Third, the consistent dominance of kNN models over longer
504 horizons underscores the increasing importance of spatial patterns or dependencies in
505 forecasting. Finally, more ML models beat the ARDI than they did ARD. This indicates
506 that the data-rich linear model, ARDI, has less accuracy than the data-poor linear model,
507 the ARD.
508 Our analysis echos the finding of Goulet Coulombe et al. (2022) that nonlinearity is
509 of vital importance in forecasting macroeconomic indicators.


510 7 ML Forecasting in Highly Volatile Environments

511 In the preceding section, we highlighted the proficiency of machine learning (ML) models
512 in exploiting non-linearities in the data. This section advances that discussion by criti-
513 cally evaluating the performance of these models during significant economic instabilities,
514 notably 1) NBER recessions and 2) the extreme fluctuations during the COVID-19 pan-
515 demic. In this section, we aim to shed light on the predictive strength and resilience of
516 ML models across real, nominal, and financial variables during such downturns.


517 7.1 Performance During NBER Recessions

518 In this subsection, we analyze the forecasting accuracy and robustness of ML models
519 during NBER recession periods. Focusing on these recession episodes allows us to equip
520 policymakers, financial analysts, and forecasters with actionable insights into which mod-
521 els have consistently outperformed the baseline under economic stress. This analysis is


    20

522 particularly vital when the likelihood of a recession is high, enabling us to develop pre-
523 emptive strategies for economic forecasting.
524 We present the results for the real variables in the right-hand side panel of table 6.
525 In the short run, AdaBoost and its DI counterpart emerge as the top performers in pre-
526 dicting these five variables, a phenomenon that can be attributed to AdaBoost’s strategy
527 of assigning higher weight to trees with larger errors and lower weight to those with
528 smaller errors. This adaptability makes AdaBoost particularly effective during periods
529 of heightened short-term fluctuations. As we extend our analysis to longer horizons, we
530 observe that Support Vector Regression (SVR) models, especially those utilizing linear
531 and sigmoid kernels, surpass the baseline predictions. For employment growth and un-
532 employment rate, the efficacy of the SVR models, particularly with the linear kernel,
533 in longer-term forecasts demonstrates their strength in capturing the underlying linear
534 trends within the economic indicators. Whereas, for real personal income and real per-
535 sonal consumption expenditure, the success of models with sigmoid kernels hints at their
536 potential to handle non-linear patterns. One notable observation across all five variables
537 is that the performance of most models relative to the baseline improves during recessions
538 compared to the full results. We can also see that the best models have lower relative
539 RMSPE during recessions across all horizons than the full sample. For example, for in-
540 dustrial production, at h = 12, the best model in the full POOS is kNN (inverse) - DI
541 with a relative RMSPE of 0.85. The relative RMSPE of this model for the same horizon
542 during recessions is 0.80, and the best model at h = 12 during recessions is SVR with
543 a linear kernel with a relative RMSPE of 0.73. We also notice that more models out-
544 perform the baseline during the recessions than the full POOS for all five variables. For
545 example, while only six models outperformed the ARD baseline for industrial production
546 at h = 12 in the full POOS, seventeen models outperform the baseline at h = 12 during
547 the recession periods. Similarly, for employment the corresponding number of models
548 beating the baseline are 4 and 18. For unemployment, the models
549 For the nominal variables detailed in table 7, we observe that machine learning mod-
550 els, particularly AdaBoost and its DI counterpart, outperform the baseline models in


    21

551 the short run. The machine learning models showed improved performance at h = 1
552 during recessions for all nominal variables compared to the full sample. However, the
553 improvement in prediction accuracy over traditional methods is not statistically signifi-
554 cant. As we look towards longer forecasting horizons, the baseline Autoregressive Direct
555 (ARD) model emerges as the most accurate predictor. The ARD model’s accuracy could
556 be attributed to the nature of inflation and its expectations, which are typically well
557 anticipated by markets and individuals alike, leading to its variations acting more like
558 exogenous noise in the economic system. Consequently, models that rely on a wealth
559 of data points tend to be overparameterized, resulting in a diminished predictive perfor-
560 mance for these nominal series.². This pattern suggests that for nominal variables like
561 inflation, simpler models that can effectively capture long-term trends without overfitting
562 to short-term fluctuations may provide more reliable forecasts.
563 Finally, the results for the financial variables are detailed in the right-hand side panel
564 of table 8. Our analysis reveals that multiple models outperform the baseline, with
565 the degree of improvement varying across different forecasting horizons. Specifically, for
566 GS1, the SVR model with an RBF kernel shows a notable improvement over the baseline,
567 enhancing predictions by 0.41 and 0.38 percentage points (pp) for horizons of 9 and 12
568 months, respectively. Similarly, for the US/UK Foreign Exchange rate, the SVR model
569 employing a sigmoid kernel surpasses the baseline’s accuracy at a 12-month horizon.
570 For the other three variables–GS10, S&P 500, and CA/US Exchange Rate–although we
571 observe a relative RMSPE of less than 1 for various models at different horizons, the
572 increase in accuracy is not statistically significant. Therefore, while machine learning
573 models can offer marginal improvements in forecasting accuracy for financial variables,
574 these gains are minimal.
575 The machine learning models continue to show improved performance during reces-
576 sions compared to the full out-of-sample forecasts for the financial variables. Notably, the
577 relative RMSPE of the best model for the full out-of-sample is less than 1 only for 4 of the
578 25 variable-horizon combinations. On the other hand, for the recession periods, the best
       2See Kotchoni et al. (2019)


    22

579 model’s relative RMSPE is less than 1 for 20 of the 25 variable-horizon combinations.
580 In conclusion, our analysis reveals the differentiated performance of machine learning
581 models across real, nominal, and financial variables during recession periods, highlighting
582 their potential in accurate forecasting under economic stress conditions. The adaptability
583 of models like AdaBoost and SVR is particularly noteworthy, suggesting their utility in
584 addressing the complex dynamics of economic downturns.
585 During recessions, the machine learning models improve upon both the baseline ARD
586 and their relative performance to baseline for the full POOS period for the real variables.
587 On the other hand, for the other two categories, machine learning models continue to show
588 smaller RMSPE relative to the baseline compared to the full POOS period. However,
589 they improve over the baseline only for a few variables at select horizons during recession
590 periods.
591 As we transition from the context of NBER recessions to the unprecedented challenges
592 posed by the COVID-19 pandemic, it is crucial to recognize the initial severe disruptions
593 it caused across key economic indicators. Specifically, the five real variables in our study
594 – industrial production, employment, real personal income, unemployment rate, and real
595 personal consumption expenditure – experienced significant spikes at the pandemic’s
596 onset which are larger in magnitude than the disruptions during the Great Recession of
597 2008.³ Given our findings that machine learning models are adept at capturing extreme
598 fluctuations, this sets a compelling premise for extending our analysis to include the data
599 during COVID and post-COVID up to the end of 2023. By incorporating data from this
600 period, we aim to validate the resilience and forecasting accuracy of these models in the
601 face of such a global crisis. The next section presents the results for the full pseudo-out-
602 of-sample forecasts up to 2023m12. These results underline the critical role of advanced
603 forecasting techniques in navigating through and beyond the economic ramifications of
604 the COVID-19 pandemic.
       3See Appendix figure 8 for more details.






    23

605 7.2 Post-Covid Data Analysis

606 In figure 1, we compare the relative RMSPE of the best models for the pre-pandemic
607 sample up to 2019m12 and the full sample up to 2023m12 for all the real variables. This
608 figure shows that the machine-learning models outperform the baseline more in the full
609 sample than during the pre-pandemic sample.
610 Further, tables 10 to 14 detail our pseudo-out-of-sample forecast results spanning from
611 January 1960 to December 2023, encompassing the entirety of the COVID-19 pandemic
612 and the subsequent two years. Several critical insights emerge from this analysis:
613 First, the baseline ARD model’s forecasting accuracy declined for all variables across
614 every horizon compared to pre-pandemic performances, with the unemployment rate’s 1-
615 month ahead prediction accuracy dropping by 16.82 pp. This decline in accuracy reduced
616 the overall statistical significance of the models’ forecasts relative to the baseline during
617 the examined period.
618 Second, despite these challenges, our analysis demonstrates the resilience of machine
619 learning models, as at least one ML model outperforms the baseline for each variable and
620 forecasting horizon. Notably, the 1-month ahead forecast for real Personal Consumption
621 Expenditures (PCE) stands out, with almost all ML models–excluding Random Forest -
622 DI, Decision Trees, and Decision Tree - DI–surpassing the baseline’s accuracy.
623 These findings highlight a generalized decrease in forecast accuracy across all models,
624 including the baseline, when factoring in the period during and after the COVID-19 pan-
625 demic. However, it is evident that certain ML models still managed to exceed baseline
626 performance in accuracy, albeit not consistently across the entire sample. This under-
627 scores the potential of machine learning approaches in adapting to and forecasting under
628 the unprecedented economic conditions introduced by the pandemic, suggesting avenues
629 for future research to refine these models for enhanced predictive performance in similarly
630 volatile contexts.







    24

631 8 Stability of Forecast Results

632 This section evaluates the stability of forecast performance for real, nominal, and financial
633 variables over time, utilizing a 36-month rolling average of the RMSPE similar to Kotchoni
634 et al. (2019). Our analysis not only assesses model adaptability under changing economic
635 conditions but also highlights the impact of major economic events on forecast accuracy.
636 Figures 2 – 4 display the 3-year moving average of the RMSPE of select models and
637 the baseline for real, nominal, and financial variables, respectively, at a forecast horizon
638 of h = 3 months. By applying a 36-month rolling average to the RMSPE, we can observe
639 how the accuracy of forecasts evolves over time, shedding light on the models’ adaptability
640 to changing economic conditions. The models we chose are AdaBoost, AdaBoost - DI,
641 kNN(uniform), kNN (uniform) - DI, and SVR (linear) - DI. We selected these models
642 based on their consistent performance, emerging as the top-performing models across
643 all fifteen variables and various forecasting horizons. For each category, we show the
644 forecasting performance of the original sample period up to 2019 in the left-hand panel.
645 On the right, we also show the extended sample that includes the COVID-19 pandemic.
646 For industrial production, which is in the first row of figure 2, SVR (linear) - DI is
647 the best performer post-1980s. In the case of real personal income, We find that kNN
648 (uniform) - DI and AdaBoost keep exchanging places for the lowest RMSPE up to 2019.
649 However, post-COVID, we also find that the baseline and AdaBoost - DI models outper-
650 form the others. The economic stability of the Great Moderation period from the mid-
651 1980s to 2007 contributed to a decline in the RMSPE of the real activity series, especially
652 employment, real personal income, unemployment rate, and real personal consumption
653 expenditure. Kotchoni et al. (2019) also find that the RMSPE lowered during the Great
654 Moderation period. The relative RMSPE also systematically decreased during and after
655 recessions. Additionally, the increase in the relative RMSPE was larger around the oil
656 price shocks (1973-1974), Great Inflation (1965-1982), Great Recession (2008-2009), and
657 COVID-19 pandemic (2020) than the increase in the relative RMSPE around the 1991
658 and 2001 recessions. These volatility changes for industrial production and employment
659 align with macroeconomic uncertainty dynamics in Jurado et al. (2015). Also, from the

        25

660 post-COVID plots, we can see that the forecasting performance of all models decreased
661 significantly, with all five variables showing an increased RMSPE. At the end of 2023, the
662 forecast accuracy started improving for all variables except real personal income. This
663 could be because of a second set of fluctuations in income in 2021 due to the checks after
664 an initial disruption in 2020. However, the RMSPE of all models is still above pre-covid
665 levels.
666 Coming to the nominal variables in figure 3, we observe that the baseline ARD model
667 consistently has the lowest RMSPE of all models throughout the sample period up to
668 2023.
669 A slow downward trend in the RMSPE that began in the early 1980s vanished at the
670 beginning of the 1990s, coinciding with the inflation-targeting regime. As suggested by
671 Clark & Davig (2009), Jørgensen & Lansing (2019), etc., improved anchoring of inflation
672 and expectations results in overall lowered volatility, which could have led to better
673 forecasting and lower RMSPE during this period. This downward trend also echoes the
674 36-month rolling RMSPE plot in Kotchoni et al. (2019). In the early 2000s, the US
675 economy saw a period of sustained high inflation due to increased economic activity
676 worldwide. During this period, the RMSPE of all the nominal variables increased and
677 peaked at the Great Recession of 2007-08.
678 Finally, two peaks emerge when we look at the finance variables in figure 4. For
679 S&P 500, GS1, GS10, and the CA/US foreign exchange rate, the first spike in RMSPE
680 happens during the 1980-82 recessions following the Iranian revolution and the tightening
681 of monetary policy. The second peak happens for GS10 and the US/UK forex rate during
682 the Great Recession. For all financial variables, the ARD model performs the best.
683 However, during the recession periods, the kNN (uniform) - DI model also had the lowest
684 RMSPE.
685 Now, we focus on the cumulative forecast errors in figures 5–7. From the forecast
686 errors, we learn about the overall bias of each model. i.e., are the models consistently
687 predicting above or below the observed value?
688 For all real variables except the change in unemployment rate in figure 5, we see that


    26

689 the benchmark ARD model has the least bias. We find that AdaBoost and AdaBoost -
690 DI produce the least biased forecasts for the unemployment rate.
691 While we know that the baseline ARD model produces the most accurate predictions
692 for all nominal variables at the aggregate level, it is not always the least biased. In fact,
693 from figure 6, we can see that multiple ML models, such as SVR (linear) - DI, AdaBoost,
694 and AdaBoost - DI, have lower bias than the baseline for CPI, CPI less food, and PCEPI.
695 A similar pattern follows for the financial variables in figure 7, where the RW has the
696 least bias only for the US/UK foreign exchange rate. In contrast, for the other variables,
697 different ML models have the least bias at different times.
698 In summary, different ML models have the lowest RMSPE over time for real variables,
699 while for nominal and financial variables, the baseline model has the lowest RMSPE. The
700 RMSPE of all models for the real variables increased in the aftermath of the COVID-19
701 pandemic. Machine learning models showed more bias than the ARD model in forecasting
702 real variables, while the opposite was true for nominal and financial variables.


703 9 Conclusion

704 In this chapter, we conducted a comprehensive examination of the forecasting abilities of
705 machine learning (ML) models in comparison with traditional econometric models across
706 a diverse set of macroeconomic variables. Our analysis, spanning a wide range of real,
707 nominal, and financial indicators, provides critical insights into the evolving landscape of
708 economic forecasting.
709 Firstly, our study reveals that ML models demonstrate superior predictive accuracy
710 for real variables, not only in the complete pre-pandemic sample but also in scenarios in-
711 volving high volatility, such as economic recessions and the COVID-19 pandemic. While
712 they may fall short in forecasting nominal and financial variables compared to tradi-
713 tional econometric benchmarks, this differential performance suggests that the inherent
714 strengths of ML models – primarily their capacity to uncover non-linear patterns – make
715 them more suited to contexts where such complexities are prevalent.


    27

716 Additionally, our study underscores the effectiveness of dimension reduction tech-
717 niques like Principal Component Analysis (PCA) in improving the performance of ML
718 models for certain variables over longer horizons. By distilling information from datasets
719 without introducing unnecessary noise, PCA proves to be a valuable tool for enhancing
720 forecasting accuracy.
721 During periods of high volatility, ML models consistently outshined benchmark models
722 in forecasting real variables, indicating their resilience and flexibility under challenging
723 environments. The machine learning models show more improvements over the baseline
724 during economic downturns compared to the overall sample.
725 In conclusion, our chapter contributes to the ongoing debate on the effectiveness of
726 machine learning in economic forecasting. Our exercise points to where these models
727 fit within the spectrum of forecasting tools and under what conditions they are most
728 effective. Looking forward, it suggests a potential avenue for further research into hy-
729 brid models that combine the strengths of econometric and machine learning methods
730 to enhance predictive performance across all economic variables. The findings also offer
731 practical implications for policymakers and practitioners in selecting appropriate fore-
732 casting models tailored to specific economic indicators and conditions.
733 Moving forward, we plan to enhance our methodology by exploring dimension reduc-
734 tion techniques, such as regularization methods like Lasso and Ridge, which can improve
735 the model’s ability to prevent overfitting and increase prediction accuracy. We also aim
736 to optimize time series models by utilizing model selection criteria such as the Akaike
737 Information Criterion (AIC) and Bayesian Information Criterion (BIC) to identify the
738 model specifications systematically. Furthermore, we need to ensure our results hold after
739 fine-tuning the hyperparameters of our machine-learning models. Through testing and
740 refinement, we seek to strengthen the reliability and accuracy of our forecasting tools so
741 they can effectively perform across diverse economic scenarios.








    28

742 References

743 Bai, J. & Ng, S. (2002). Determining the number of factors in approximate factor models.
744 Econometrica, 70(1), 191–221.

745 Breiman, L. (1996). Bagging predictors. Machine learning, 24, 123–140.

746 Breiman, L. (2001). Random forests. Machine learning, 45, 5–32.

747 Chen, T. & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceed-
748 ings of the 22nd acm sigkdd international conference on knowledge discovery and data
749 mining (pp. 785–794).

750 Choudhary, M. A. & Haider, A. (2012). Neural network models for inflation forecasting:
751 an appraisal. Applied Economics, 44(20), 2631–2635.

752 Clark, T. E. & Davig, T. (2009). The relationship between inflation and inflation expec-
753 tations. memo to the FOMC, November, 30.

754 Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accu racy. Journal of
755 Business and Economic Statistics, 13(3), 253–263.

756 Dietterich, T. G. (2000). Ensemble methods in machine learning. In International work-
757 shop on multiple classifier systems (pp. 1–15).: Springer.

758 Freund, Y. & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learn-
759 ing and an application to boosting. Journal of computer and system sciences, 55(1),
760 119–139.

761 Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine.
762 Annals of statistics, (pp. 1189–1232).

763 Goulet Coulombe, P., Leroux, M., Stevanovic, D., & Surprenant, S. (2022). How is ma-
764 chine learning useful for macroeconomic forecasting? Journal of Applied Econometrics,
765 37(5), 920–964.



    29

766 Jørgensen, P. L. & Lansing, K. J. (2019). Anchored inflation expectations and the slope
767 of the phillips curve. Federal Reserve Bank of San Francisco Working Paper, 27.

768 Jurado, K., Ludvigson, S. C., & Ng, S. (2015). Measuring uncertainty. American Eco-
769 nomic Review, 105(3), 1177–1216.

770 Kauffman, R. G. (1999). Indicator qualities of the napm report on business®. Journal
771 of Supply Chain Management, 35(1), 29–37.

772 Kotchoni, R., Leroux, M., & Stevanovic, D. (2019). Macroeconomic forecast accuracy in
773 a data-rich environment. Journal of Applied Econometrics, 34(7), 1050–1072.

774 Ma, Y. & Zhu, L. (2013). A review on dimension reduction. International Statistical
775 Review, 81(1), 134–150.

776 McCracken, M. W. & Ng, S. (2016). Fred-md: A monthly database for macroeconomic
777 research. Journal of Business & Economic Statistics, 34(4), 574–589.

778 Milunovich, G. (2020). Forecasting australia’s real house price index: A comparison of
779 time series and machine learning methods. Journal of Forecasting, 39(7), 1098–1118.

780 Mullainathan, S. & Spiess, J. (2017). Machine learning: an applied econometric approach.
781 Journal of Economic Perspectives, 31(2), 87–106.

782 Quinlan, J. R. (1986). Induction of decision trees. Machine learning, 1, 81–106.

783 Serrano, C. & Hoesli, M. (2007). Forecasting ereit returns. Journal of Real Estate
784 Portfolio Management, 13(4), 293–310.

785 Stock, J. H. & Watson, M. W. (2002). Macroeconomic forecasting using diffusion indexes.
786 Journal of Business & Economic Statistics, 20(2), 147–162.










    30

    Table 1: List of all forecasting models

                            Model list
Baseline models
ARD                     Autoregressive direct
RW                      Random walk
Individual machine learning models
kNN(uniform)            K-nearest neighbor (uniform weighted) regression
kNN(inverse)            K-nearest neighbor (inverse weighted) regression
Decision Tree           Decision tree regression
SVR (linear)            Support vector regression (linear kernel)
SVR (polynomial)        Support vector regression (polynomial kernel)
SVR (rbf)               Support vector regression (rbf kernel)
SVR (sigmoid)           Support vector regression (sigmoid kernel)
Ensemble machine learning models
Random forest           Random forest regression
XGBoost                 XGBoost regression
AdaBoost                AdaBoost regression
Gradient Boost          Gradient boost regression
Individual machine learning models using dimension reduction
kNN(uniform) - DI       K-nearest neighbor (uniform weighted) regression with diffusion index
kNN(inverse) - DI       K-nearest neighbor (inverse weighted) regression with diffusion index
Decision Tree - DI      Decision tree regression with diffusion index
SVR (linear) - DI       Support vector regression (linear kernel) with diffusion index
SVR (polynomial) - DI   Support vector regression (polynomial kernel) with diffusion index
SVR (rbf) - DI          Support vector regression (rbf kernel) with diffusion index
SVR (sigmoid) - DI      Support vector regression (sigmoid kernel) with diffusion index
Ensemble machine learning models using dimension reduction
Random Forest - DI      Random forest regression with diffusion index
XGBoost - DI            XGBoost regression with diffusion index
AdaBoost - DI           AdaBoost regression with diffusion index
Gradient Boost - DI     Gradient boost regression with diffusion index










                        31

32










Table 2: Industrial Production growth: relative RMSPE (sample period: 1960m1-2019m12)

                             Pre-Pandemic Sample               NBER recession periods
Model                  h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9     h=12
Baseline Model
ARD (RMSPE)           0.082  0.061  0.057  0.052  0.048   0.14  0.111  0.101  0.091   0.084
Individual machine learning models
kNN (uniform)          0.99   0.98   0.93   0.95   0.99   0.98   1.06   1.06   1.09    1.06
kNN (inverse)          0.99   0.98   0.93   0.95   0.99   0.96   1.06   1.06   1.09    1.05
Decision Tree          1.32   1.26   1.20   1.12   1.11   1.23   1.07   1.04   0.89   0.85*
SVR (linear)           1.16   1.15   1.05   1.04   1.11   1.00   1.00   0.87  0.83*  0.73***
SVR (polynomial)       1.03   1.14   1.19   1.15   1.11   1.06   1.16   1.09   1.09    1.07
SVR (rbf)              1.01   1.00   0.93   0.94   0.99   1.04   1.08   1.00   0.96    0.94
SVR (sigmoid)          1.05   1.03   0.98   0.99   1.07   1.01   0.92   0.92   0.88  0.82***
Ensemble machine learning models
Random Forest          0.97   1.03   1.03   1.02   1.00   0.93   0.90   0.96   0.86   0.78**
XGBoost                1.01   1.00   0.98   1.01   1.01   0.99   0.92   0.88   0.89   0.77*
AdaBoost              0.95**  0.95   1.00   0.99   0.96   0.95   0.89   0.99   0.89   0.76**
Gradient Boost         1.25   1.22   1.19   1.10   1.10   1.13   1.03   1.03  0.84*   0.81*
Individual machine learning models using dimension reduction
kNN (uniform) - DI    0.97*   0.94  0.87*   0.84   0.85  0.93*   0.90   0.88  0.82*   0.81**
kNN (inverse) - DI    0.96*   0.94  0.88*   0.84   0.85  0.92**  0.90   0.88  0.82*   0.80**
Decision Tree - DI     1.22   1.22   1.25   1.23   1.21   1.07   0.95   0.95   1.00    0.97
SVR (linear) - DI      1.12   1.39   1.33   1.23   1.35   1.05   1.11   1.00   0.99    1.00
SVR (polynomial) - DI  1.91   1.50   1.07   1.14   1.22   2.88   2.06   1.23   0.82    0.86
SVR (rbf) - DI         1.06   1.11   1.01   0.97   1.01   1.17   1.30   1.18   1.03    0.95
SVR (sigmoid) - DI     1.27   1.26   1.19   1.35   1.36   1.27   1.09   1.01   1.22    1.14
Ensemble machine learning models using dimension reduction
Random Forest - DI     1.21   1.25   1.20   1.22   1.20   1.08   0.97   0.97   0.98    0.97
XGBoost - DI           1.13   1.01   0.99   1.02   1.09   1.22   1.01   0.98   0.86    0.83
AdaBoost - DI      0.97*      0.95   0.98   1.00   1.04   0.95   0.89   0.90  0.81*   0.80*
Gradient Boost - DI    1.22   1.23   1.19   1.19   1.18   1.15   0.98   1.00   0.95    0.93

33










    Table 3: Employment: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample        NBER recession periods
Model                  h=1     h=3    h=6    h=9    h=12   h=1    h=3    h=6      h=9      h=12
Baseline Model
ARD (RMSPE)           0.018   0.014  0.015  0.015  0.016  0.024  0.025  0.026    0.026    0.026
Individual machine learning models
kNN (uniform)          1.04    1.05   1.03   1.04   1.05   1.28   1.25   1.23     1.19     1.12
kNN (inverse)          1.03    1.04   1.02   1.03   1.04   1.25   1.24   1.23     1.19     1.11
Decision Tree          1.24    1.24   1.32   1.24   1.15   1.32   1.19   1.28     1.00     0.86
SVR (linear)           1.11    1.10   1.07   1.05   1.05   1.13   1.02   0.88   0.72***  0.59***
SVR (polynomial)       1.22    1.49   1.49   1.40   1.33   1.73   1.64   1.39     1.28     1.19
SVR (rbf)              1.07    1.12   1.07   1.06   1.06   1.43   1.39   1.16     0.97     0.86
SVR (sigmoid)          1.07    1.08   1.09   1.04   1.05   1.14   1.01   0.96    0.78**  0.66***
Ensemble machine learning models
Random Forest         0.92**   0.97   1.08   1.04   1.04   0.93   1.00   1.15     0.92    0.81*
XGBoost                0.95    0.93   0.99   0.98   1.01   1.00   1.02   1.08     0.94     0.83
AdaBoost              0.92**   0.92   1.06   1.07   1.05   0.99   1.03   1.16     0.97     0.83
Gradient Boost         1.20    1.14   1.30   1.20   1.16   1.25   1.08   1.29     0.96     0.89
Individual machine learning models using dimension reduction
kNN (uniform) - DI     0.98    0.97   0.94   0.94   0.93   1.12   1.12   1.03     0.89     0.79
kNN (inverse) - DI     0.98    0.97   0.94   0.94   0.93   1.11   1.11   1.03     0.89     0.79
Decision Tree - DI     1.21    1.15   1.23   1.19   1.21   1.06   1.12   1.13     1.01     0.95
SVR (linear) - DI      0.99   0.87**  0.90   0.96   1.05   0.94 0.82*   0.80**  0.73***  0.57***
SVR (polynomial) - DI  3.05    4.10   4.18   3.79   3.43   6.04   6.70   5.73     3.01     1.00
SVR (rbf) - DI         1.07    1.13   1.08   1.07   1.08   1.51   1.49   1.25     1.00     0.87
SVR (sigmoid) - DI     3.62    3.55   2.89   2.61   2.41   6.24   4.80   3.26     2.01     1.77
Ensemble machine learning models using dimension reduction
Random Forest - DI     1.23    1.15   1.22   1.18   1.19   1.05   1.09   1.12     1.00     0.93
XGBoost - DI           0.97    0.91   0.92   0.96   0.98   0.93   0.93   0.96    0.88*    0.81*
AdaBoost - DI         0.92**  0.89*   0.94   0.96   0.98   0.90   0.91   0.99     0.93    0.83*
Gradient Boost - DI    1.17    1.10   1.21   1.18   1.18   1.07   1.07   1.10     1.00     0.94

34










    Table 4: CPI inflation: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                        NBER recession periods
Model                  h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9   h=12
Baseline Model
ARD (RMSPE)           0.031  0.027  0.023  0.021  0.021  0.052  0.048  0.036  0.032  0.03
Individual machine learning models
kNN (uniform)          1.11   1.27   1.45   1.53   1.53   0.94   1.14   1.64   1.39  1.38
kNN (inverse)          1.12   1.27   1.45   1.52   1.53   0.94   1.14   1.64   1.38  1.37
Decision Tree          1.39   1.67   1.66   1.65   1.66   1.25   1.65   1.50   1.27  1.35
SVR (linear)           1.29   1.38   1.58   1.58   1.54   1.06   1.07   1.42   1.45  1.49
SVR (polynomial)       1.10   1.33   1.49   1.50   1.50   0.99   1.31   1.53   1.32  1.36
SVR (rbf)              1.10   1.23   1.39   1.47   1.47   0.94   1.11   1.57   1.34  1.35
SVR (sigmoid)          1.17   1.31   1.48   1.57   1.52   1.00   1.10   1.53   1.42  1.38
Ensemble machine learning models
Random Forest          1.12   1.27   1.36   1.34   1.33   1.02   1.32   1.44   1.16  1.22
XGBoost                1.21   1.25   1.29   1.31   1.32   1.07   1.17   1.33   1.20  1.24
AdaBoost               1.08   1.21   1.26   1.28   1.30   0.94   1.21   1.41   1.18  1.18
Gradient Boost         1.34   1.61   1.54   1.57   1.63   1.17   1.67   1.55   1.30  1.35
Individual machine learning models using dimension reduction
kNN (uniform) - DI     1.12   1.28   1.43   1.47   1.46   0.96   1.17   1.61   1.29  1.25
kNN (inverse) - DI     1.12   1.28   1.43   1.47   1.46   0.96   1.17   1.61   1.29  1.24
Decision Tree - DI     1.34   1.37   1.49   1.63   1.60   1.11   1.12   1.40   1.36  1.21
SVR (linear) - DI      1.38   1.63   2.09   2.19   2.08   1.26   1.51   2.29   2.08  1.87
SVR (polynomial) - DI  2.11   2.44   2.76   3.14   3.08   1.59   1.56   2.37   3.09  5.00
SVR (rbf) - DI         1.14   1.29   1.53   1.63   1.62   0.96   1.14   1.66   1.41  1.35
SVR (sigmoid) - DI     1.38   1.60   1.88   2.04   1.96   1.20   1.47   2.02   1.90  1.68
Ensemble machine learning models using dimension reduction
Random Forest - DI     1.35   1.38   1.50   1.66   1.58   1.13   1.12   1.41   1.39  1.20
XGBoost - DI           1.21   1.19   1.32   1.38   1.38   1.13   1.08   1.35   1.22  1.14
AdaBoost - DI          1.06   1.14   1.23   1.28   1.30   0.88   1.01   1.34   1.16  1.18
Gradient Boost - DI    1.31   1.35   1.40   1.62   1.54   1.14   1.10   1.36   1.30  1.11

35










    Table 5: S&P 500: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                        NBER recession periods
Model                  h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9    h=12
Baseline Model
RW (RMSPE)            0.438  0.296  0.228  0.193  0.172  0.735  0.489  0.341  0.281  0.241
Individual machine learning models
kNN (uniform)          1.01   0.99   0.99   1.00   1.00   1.00   1.02   1.10   1.16   1.22
kNN (inverse)          1.01   0.99   0.99   1.00   1.00   1.00   1.02   1.10   1.16   1.21
Decision Tree          1.32   1.37   1.30   1.27   1.15   1.20   1.32   1.12   1.11   1.02
SVR (linear)           1.22   1.30   1.32   1.34   1.35   0.96   1.08   0.98   0.96   1.08
SVR (polynomial)       1.02   1.19   1.37   1.26   1.20   1.07   1.23   1.07   1.18   1.25
SVR (rbf)              0.99   1.01   1.02   1.06   1.08   1.00   1.03   1.05   1.14   1.22
SVR (sigmoid)          1.03   1.10   1.14   1.19   1.27   0.94   1.02   0.99   1.03   1.15
Ensemble machine learning models
Random Forest          1.02   1.11   1.09   1.12   1.09   1.01   1.17   1.01   1.03   1.05
XGBoost                1.07   1.09   1.11   1.14   1.12   1.05   1.14   0.99   1.06   1.09
AdaBoost               0.98   1.03   1.04   1.03   1.06   0.97   1.06   0.98   0.99   1.06
Gradient Boost         1.25   1.34   1.24   1.26   1.15   1.16   1.31   1.07   1.11   1.03
Individual machine learning models using dimension reduction
kNN (uniform) - DI     1.02   1.06   1.05   1.03   1.04   1.02   1.04   0.97   1.06   1.15
kNN (inverse) - DI     1.02   1.06   1.06   1.03   1.04   1.01   1.04   0.97   1.06   1.15
Decision Tree - DI     1.29   1.44   1.43   1.33   1.29   1.07   1.30   1.19   1.11   1.21
SVR (linear) - DI      1.05   1.23   1.39   1.54   1.68   1.02   1.13   1.14   1.16   1.28
SVR (polynomial) - DI  1.12   1.10   1.16   1.23   1.38   1.25   1.23   1.05   1.02   1.13
SVR (rbf) - DI         1.02   1.07   1.08   1.12   1.15   1.04   1.10   1.16   1.29   1.34
SVR (sigmoid) - DI     1.10   1.15   1.25   1.28   1.30   1.13   1.22   1.15   1.20   1.35
Ensemble machine learning models using dimension reduction
Random Forest - DI     1.29   1.43   1.41   1.29   1.28   1.03   1.30   1.18   1.12   1.20
XGBoost - DI           1.14   1.18   1.28   1.22   1.20   1.05   1.22   1.17   1.11   1.21
AdaBoost - DI          0.99   1.05   1.13   1.10   1.11   0.97   1.05   1.04   1.00   1.13
Gradient Boost - DI    1.22   1.36   1.35   1.25   1.28   1.02   1.25   1.16   1.11   1.20

36










    Table 6: Relative RMSPE of the best models for real variables (sample period: 1960m1-2019m12)

                             Pre-Pandemic Sample                                         NBER recession periods
Model               h=1      h=3      h=6    h=9   h=12                       h=1     h=3         h=6     h=9      h=12
Industrial Production
Baseline - ARD  0.082       0.061   0.057  0.052  0.047    Baseline - ARD     0.14  0.111        0.101   0.091    0.084

kNN (inverse)       0.99     1.00    0.92   0.90   0.93    SVR (linear)       1.00   1.00         0.87   0.83*   0.73***
AdaBoost      0.95***        0.94    1.00   1.00   0.96    AdaBoost           0.94  0.88*         0.99    0.88    0.76**
kNN (uniform) - DI  1.02     0.96   0.87*  0.89*   0.94    Random Forest      0.95   0.89         0.96    0.86    0.78**
kNN (inverse) - DI  1.02     0.96   0.88*  0.89*   0.94    Gradient Boost     1.12   1.02         1.03   0.84*    0.82*
AdaBoost - DI      0.97*     0.96    0.99   1.00   1.05    AdaBoost - DI      0.96   0.89         0.90   0.81*    0.79*
Employment
Baseline - ARD     0.018    0.014   0.015  0.015  0.016    Baseline - ARD    0.024  0.025  0.026         0.026    0.026

Random Forest      0.91**    0.98    1.08   1.04   1.04    SVR (linear)       1.13   1.02         0.88  0.72***  0.59***
AdaBoost           0.92**    0.92    1.06   1.08   1.06    SVR (sigmoid)      1.14   1.01         0.96   0.78**  0.66***
kNN (uniform) - DI  0.98     0.91    0.87   0.89   0.92    AdaBoost           1.00   1.03         1.15    0.96    0.82*
SVR (linear) - DI   0.99    0.87**   0.90   0.96   1.05    SVR (linear) - DI  0.94  0.82*  0.80**       0.73***  0.57***
AdaBoost - DI      0.93**   0.90*    0.93   0.96   0.99    AdaBoost - DI     0.89*   0.92         0.99    0.92    0.84*
Real Personal Income
Baseline - ARD     0.075    0.038   0.027  0.024  0.022    Baseline - ARD    0.105  0.061        0.045 0.039      0.037

SVR (rbf)          0.92**    0.95   0.94*   0.96   0.98    SVR (rbf)          0.91   0.94        0.92**   0.95     0.96
SVR (sigmoid)      0.94*     0.97    0.96   1.03   1.08    SVR (sigmoid)      0.90 0.88**        0.80**   0.83    0.84*
AdaBoost           0.92**    0.95  0.89***  0.92   0.93    AdaBoost           0.92   0.96        0.84**   0.87   0.88***
SVR (rbf) - DI     0.93*     0.99    1.00   1.02   1.05    SVR (linear)       1.07   1.18   0.93          0.80    0.84*
AdaBoost - DI    0.99        0.95    0.95   0.94   0.95    AdaBoost - DI     0.86*   0.95        0.92*    0.89     0.87
Unemployment Rate
Baseline - ARD  2.011       1.329   1.212  1.151  1.117    Baseline - ARD    2.711  2.204        1.999   1.846    1.672

kNN (inverse)       0.98     0.95    0.88  0.87*  0.87*    SVR (linear)       1.01   0.91        0.83*  0.71***  0.56***
AdaBoost      0.92***       0.89**   0.92   0.91   0.90    AdaBoost         0.89***  0.92         1.03    0.86    0.74**
kNN (uniform) - DI  1.01    0.90*   0.84*  0.87*  0.89**   XGBoost           0.85**  0.88**       0.99    0.87    0.75**
kNN (inverse) - DI  1.01    0.90*   0.84*  0.86*  0.88**   XGBoost - DI       1.01   0.78**       0.82   0.73**   0.68**
AdaBoost - DI     0.96**    0.87**  0.83**  0.88   0.94    AdaBoost - DI      0.94   0.83**       0.86   0.79**   0.75**
Real Personal Consumption Expenditure
Baseline - ARD   0.06        0.03   0.022   0.02  0.019    Baseline - ARD    0.086  0.053         0.04   0.038    0.037

SVR (rbf)          0.97**  0.92***   0.95   0.99   1.01    SVR (linear)       1.13   0.93         0.86   0.77**    0.88
SVR (sigmoid)       1.05     0.96    0.99   1.01   1.04    SVR (sigmoid)      1.01  0.81*        0.88*   0.80*     0.86
Random Forest       1.00     0.96    0.96   0.96   1.00    Random Forest     0.92*   0.82**      0.81*    0.85     0.94
AdaBoost            0.98    0.94**  0.93*   0.99   1.00    AdaBoost           0.93  0.85***      0.88*    0.92     0.94
AdaBoost - DI       0.98    0.94*    0.95   0.99   1.01    AdaBoost - DI     0.91*   0.89**       0.89    0.91     1.00

37










Table 7: Relative RMSPE of the best models for nominal variables (sample period: 1960m1-2019m12)

                           Pre-Pandemic Sample               NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12                         h=1    h=3    h=6     h=9    h=12
CPI: All Items
Baseline - ARD      0.031  0.027  0.023  0.021  0.021    Baseline - ARD      0.052  0.048  0.036   0.032   0.03

kNN (uniform)        1.12   1.27   1.44   1.51   1.52    kNN (uniform)        0.96   1.14   1.59    1.35   1.36
Random Forest        1.12   1.27   1.37   1.33   1.33    Random Forest        1.02   1.32   1.46    1.15   1.20
AdaBoost             1.07   1.20   1.25   1.29   1.31    AdaBoost             0.92   1.20   1.39    1.19   1.20
AdaBoost - DI        1.05   1.14   1.23   1.28   1.29    AdaBoost - DI        0.89   1.01   1.33    1.16   1.17
Gradient Boost - DI  1.32   1.36   1.41   1.62   1.52    Gradient Boost - DI  1.12   1.10   1.35    1.30   1.09
CPI: All Items Less Food
Baseline - ARD      0.034   0.03  0.026  0.025  0.024    Baseline - ARD      0.057  0.055  0.043   0.039  0.038

XGBoost              1.14   1.22   1.25   1.28   1.32    XGBoost              1.05   1.18   1.26    0.97   1.04
AdaBoost             1.06   1.19   1.24   1.26   1.29    AdaBoost             0.96   1.19   1.32    1.02   0.98
Decision Tree - DI   1.29   1.34   1.47   1.59   1.55    Decision Tree - DI   0.95   1.16   1.45    1.24   1.19
XGBoost - DI         1.11   1.14   1.29   1.32   1.36    XGBoost - DI         0.98   1.04   1.25    1.06   0.97
AdaBoost - DI        1.02   1.10   1.19   1.24   1.26    AdaBoost - DI        0.89   0.99   1.21    1.01   0.99
PCEPI
Baseline - ARD      0.022  0.019  0.017  0.016  0.016    Baseline - ARD      0.036  0.034  0.028   0.027  0.026

SVR (rbf)            1.07   1.22   1.35   1.38   1.39    SVR (rbf)            0.95   1.08   1.37    1.15   1.19
Random Forest        1.13   1.28   1.34   1.29   1.29    Random Forest        1.09   1.29   1.25    1.03   1.06
AdaBoost             1.06   1.19   1.24   1.24   1.27    AdaBoost             0.98   1.12   1.23    1.02   1.07
AdaBoost - DI        1.04   1.12   1.22   1.24   1.27    AdaBoost - DI        0.93   0.97   1.21    1.01   1.10
Gradient Boost - DI  1.25   1.26   1.49   1.62   1.64    Gradient Boost - DI  1.07   0.93   1.45    1.24   1.31
PPI: Finished Consumer Goods
Baseline - ARD      0.093   0.07   0.06  0.055  0.053    Baseline - ARD      0.148  0.118  0.092    0.08  0.065

Random Forest        1.15   1.25   1.42   1.47   1.44    Random Forest        1.04   1.18   1.51    1.33   1.51
AdaBoost             1.11   1.23   1.36   1.42   1.45    AdaBoost             0.96   1.12   1.50    1.37   1.57
SVR (rbf) - DI       1.22   1.40   1.67   1.78   1.83    SVR (rbf) - DI       0.98   1.19   1.69    1.65   1.92
XGBoost - DI         1.25   1.35   1.41   1.50   1.50    XGBoost - DI         1.15   1.34   1.40    1.41   1.50
AdaBoost - DI        1.11   1.22   1.36   1.43   1.45    AdaBoost - DI        0.97   1.12   1.46    1.38   1.58
PPI: Crude Metals
Baseline - ARD      0.457  0.321  0.284  0.267  0.256    Baseline - ARD      0.681  0.552   0.51   0.458  0.362

SVR (rbf)            1.16   1.39   1.56   1.64   1.68    SVR (rbf)            0.91   1.21   1.50    1.55   1.66
Random Forest        1.21   1.30   1.51   1.56   1.54    Random Forest        0.98   1.20   1.50    1.38   1.41
AdaBoost             1.14   1.29   1.43   1.48   1.50    AdaBoost             0.91   1.22   1.42    1.40   1.43
SVR (rbf) - DI       1.20   1.43   1.65   1.77   1.83    SVR (rbf) - DI       0.88   1.22   1.56    1.63   1.80
AdaBoost - DI        1.15   1.27   1.38   1.45   1.49    AdaBoost - DI        0.90   1.18   1.41    1.43   1.49

38










Table 8: Relative RMSPE of the best models for financial variables (sample period: 1960m1-2019m12)

                        Pre-Pandemic Sample                                         NBER recession periods
Model                h=1     h=3     h=6    h=9    h=12                       h=1     h=3    h=6    h=9    h=12
S&P 500
Baseline - RW       0.438   0.296   0.228  0.193  0.172   Baseline - RW      0.735   0.489  0.341  0.281  0.241

kNN (uniform)        1.01    0.99    0.99   1.00   1.00   SVR (linear)        0.96    1.08   0.98   0.96   1.08
kNN (inverse)        1.01    0.99    0.99   1.00   1.00   SVR (sigmoid)       0.94    1.02   0.99   1.03   1.15
SVR (rbf)            0.99    1.01    1.02   1.06   1.08   AdaBoost            0.97    1.06   0.98   0.99   1.06
AdaBoost             0.98    1.03    1.04   1.03   1.06   kNN (uniform)-DI    1.02    1.04   0.97   1.06   1.15
AdaBoost-DI          0.99    1.05    1.13   1.10   1.11   AdaBoost-DI         0.97    1.05   1.04   1.00   1.13
1-Year Treasury Rate
Baseline - RW       5.222   3.623   2.522  1.972  1.759   Baseline - RW      10.813  7.287  4.084  2.936  2.544

SVR (rbf)            1.01    1.01    1.01   1.05   1.12   SVR (rbf)           1.00    1.00   0.98   0.94   0.91
SVR (sigmoid)        1.08    1.11    1.11   1.22   1.33   SVR (sigmoid)       1.01    0.97   1.04   1.05   0.98
AdaBoost             0.98    1.04    1.06   1.17   1.30   AdaBoost            0.99    0.99   0.99   1.26   1.21
SVR (sigmoid) - DI   1.20    1.28    1.37   1.39   1.40   SVR (sigmoid) - DI  1.10    1.11   0.98   1.16   1.02
AdaBoost - DI        1.03    1.03    1.02   1.06   1.17   AdaBoost - DI       0.98    1.01   0.94   0.97   1.02
10-Year Treasury Rate
Baseline - RW       3.518   2.366   1.684  1.361  1.209   Baseline - RW      6.065   3.918  2.179  1.683   1.47

SVR (polynomial)     1.04    1.10    1.11   1.14   1.19   SVR (polynomial)    1.08    1.11   1.01   1.02   1.01
SVR (rbf)            1.03    1.05    1.08   1.14   1.18   SVR (rbf)           1.03    1.05   1.07   1.05   0.99
SVR (sigmoid)        1.11    1.13    1.20   1.29   1.35   SVR (sigmoid)       1.14    1.14   1.15   1.11   0.99
AdaBoost             1.01    1.08    1.13   1.25   1.34   AdaBoost            1.03    1.08   1.05   1.42   1.28
kNN (uniform) - DI   1.03    1.11    1.13   1.15   1.18   kNN (uniform) - DI  1.00    1.08   1.07   1.02   1.02
US/UK Foreign Exchange Rate
Baseline - RW       0.278   0.195   0.147  0.121  0.104   Baseline - RW       0.33   0.256  0.207  0.176  0.151

SVR (rbf)            1.03    1.06    1.07   1.07   1.04   SVR (sigmoid)       1.20    1.12   1.12   1.02  0.88*
kNN (uniform) - DI   1.03    1.03    1.04   1.04   1.08   kNN (uniform) - DI  0.99    1.00   0.98   0.97   0.97
kNN (inverse) - DI   1.03    1.04    1.05   1.04   1.08   kNN (inverse) - DI  0.99    0.99   0.98   0.97   0.97
SVR (rbf) - DI       1.06    1.09    1.10   1.11   1.16   XGBoost - DI        0.98    1.04   1.06   1.12   1.08
AdaBoost - DI        1.00    1.11    1.15   1.16   1.16   AdaBoost - DI      0.89**   1.06   1.03   1.07   1.04
CA/US Foreign Exchange Rate
Baseline - RW       0.174   0.117   0.087  0.073  0.064   Baseline - RW      0.236   0.165  0.118  0.095  0.081

Random Forest        1.07    1.21    1.26   1.25   1.26   Random Forest       1.15    1.35   1.11   0.93   0.93
AdaBoost             1.01    1.13    1.20   1.20   1.29   AdaBoost            1.00    1.25   1.00   1.01   0.98
SVR (rbf)            1.02    1.05    1.09   1.10   1.11   XGBoost - DI        1.13    1.32   1.13   1.10   0.93
kNN (inverse) - DI   1.05    1.11    1.14   1.15   1.21   kNN (inverse) - DI  0.98    1.00   1.13   1.16   1.07
AdaBoost - DI        1.01    1.12    1.22   1.18   1.18   AdaBoost - DI       0.93    1.15   1.22   1.21   1.09

39










                Table 9: Real Variables: RMSPE relative to ARDI (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                                         NBER recession periods
Model                h=1      h=3      h=6     h=9    h=12                        h=1     h=3    h=6    h=9    h=12
Industrial Production
Baseline - ARDI     0.089    0.065    0.058   0.057  0.061   Baseline - ARDI     0.134   0.087  0.075  0.067  0.058

kNN (uniform)       0.92**    0.93     0.89    0.83   0.72   kNN (uniform)        1.00    1.26   1.28   1.24   1.36
kNN (inverse)       0.91**    0.93     0.89    0.83   0.72   kNN (inverse)        1.00    1.26   1.28   1.24   1.36
AdaBoost           0.87*** 0.88**      0.97    0.91   0.75   Random Forest        0.99    1.13   1.29   1.17   1.13
kNN (uniform) - DI   0.94     0.90     0.85    0.82   0.73   AdaBoost             0.98    1.12   1.34   1.20   1.11
kNN (inverse) - DI   0.94     0.90     0.85    0.82   0.73   AdaBoost - DI        1.00    1.13   1.22   1.11   1.14
Employment
Baseline - ARDI     0.019 0.015       0.015   0.016  0.017   Baseline - ARDI    0.026     0.02  0.019  0.016  0.012

Random Forest      0.83***    0.97     1.05    1.02   0.97   Random Forest        0.87    1.23   1.51   1.53   1.71
XGBoost            0.86***    0.91     0.96    0.97   0.94   XGBoost              0.92    1.23   1.44   1.57   1.77
AdaBoost           0.84***    0.91     1.03    1.06   0.98   SVR (linear) - DI   0.86**   0.99   1.06   1.22   1.21
kNN (uniform) - DI  0.90**    0.90     0.84    0.88   0.85   XGBoost - DI        0.86*    1.12   1.28   1.48   1.73
SVR (linear) - DI  0.90***   0.85**    0.88    0.95   0.97   AdaBoost - DI       0.83**   1.11   1.32   1.54   1.79
Real Personal Income
Baseline - ARDI     0.091    0.044    0.029   0.027  0.028   Baseline - ARDI     0.128   0.065   0.04  0.032   0.03

SVR (rbf)          0.76***  0.82***  0.86***   0.84   0.78   SVR (linear)       0.89      1.10   1.06   0.97   1.05
SVR (sigmoid)      0.77***  0.84***   0.89**   0.90   0.85   SVR (sigmoid)       0.75**   0.83   0.91   1.00   1.05
AdaBoost           0.76***  0.82***  0.82***  0.80*   0.73   AdaBoost            0.76**   0.90   0.95   1.05   1.11
SVR (rbf) - DI     0.77***  0.85***    0.92    0.89   0.83   SVR (rbf) - DI     0.74***   0.91   1.10   1.26   1.35
AdaBoost - DI      0.82***  0.83***   0.88**   0.82   0.75   AdaBoost - DI      0.71***   0.89   1.04   1.08   1.10
Unemployment Rate
Baseline - ARDI     2.154    1.311    1.143   1.118  1.161   Baseline - ARDI     2.827   1.761  1.228  1.025  0.861

kNN (inverse)      0.91***    0.96     0.94    0.90   0.84   SVR (linear)       0.97      1.14   1.35   1.27   1.09
SVR (rbf)           0.91**    0.96     0.97    0.96   0.90   Random Forest       0.82*    1.14   1.67   1.52   1.37
AdaBoost           0.86***    0.90     0.97    0.94   0.86   XGBoost             0.82*    1.10   1.62   1.57   1.45
kNN (inverse) - DI   0.95     0.91     0.89    0.89   0.85   AdaBoost            0.85*    1.16   1.68   1.55   1.43
AdaBoost - DI      0.90***   0.89**    0.88   0.90**  0.91   XGBoost - DI         0.97    0.98   1.33   1.31   1.32
Real Personal Consumption Expenditure
Baseline - ARDI      0.07    0.034    0.025   0.026  0.025   Baseline - ARDI      0.11   0.054  0.035  0.033  0.032

kNN (uniform)      0.85***   0.86*     0.88    0.78   0.80   SVR (linear)         0.88    0.91   0.97   0.88   1.02
SVR (rbf)          0.84***   0.82**    0.82    0.76   0.77   SVR (rbf)          0.72***   0.89   1.09   1.15   1.19
Random Forest      0.87***   0.86**    0.83    0.74   0.76   SVR (sigmoid)       0.78**  0.80*   0.99   0.92   0.99
AdaBoost           0.84***   0.84**   0.80*    0.76   0.76   Random Forest      0.72***   0.81   0.92   0.98   1.10
AdaBoost - DI      0.84***   0.84**    0.82    0.76   0.77   AdaBoost - DI      0.71***   0.87   1.01   1.04   1.16

40










Table 10: Industrial Production Growth: relative RMSPE (sample period: 1960m1-2023m12)

                            Full out-of-sample        NBER recession periods
Model                   h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9    h=12
Baseline Model
ARD (RMSPE)             0.17  0.098   0.07   0.06  0.054  0.245  0.139  0.109  0.095  0.086
Individual machine learning models
kNN (uniform)           0.69   0.82   0.85   0.86   0.86   0.99   0.99   0.97   0.94   0.96
kNN (inverse)           0.69   0.82   0.85   0.86   0.86   0.99   0.99   0.97   0.94   0.96
Decision Tree           0.92   1.15   1.10   1.05   1.03   1.06   1.05   1.01   0.91  0.85*
SVR (linear)            0.88   1.18   1.09   1.08   1.06   1.01   1.00   0.91 0.85*  0.77***
SVR (polynomial)        0.73   0.88   1.06   1.05   1.01   1.03   1.12   1.08   1.09   1.07
SVR (rbf)               0.68   0.79   0.84   0.88   0.92   1.02   1.05   1.01   0.98   0.95
SVR (sigmoid)           0.74   0.85   0.91   0.93   0.99   1.01   0.96   0.94   0.89  0.86**
Ensemble machine learning models
Random Forest           0.75   0.96   0.97   0.96   0.93  0.96**  0.94   0.98   0.88  0.80**
XGBoost                 0.87   1.06   1.07   0.97   0.94   1.02   0.95   0.91   0.90  0.81*
AdaBoost                0.68   0.99   0.93   0.93   0.89   0.98   0.93   0.99   0.91  0.78**
Gradient Boost          0.83   1.09   1.08   1.06   1.01   0.95   1.03   1.00   0.89  0.85*
Individual machine learning models using dimension reduction
kNN (uniform) - DI      0.70   0.81   0.81  0.84*   0.87   1.02   1.04   1.00   0.97   0.93
kNN (inverse) - DI      0.70   0.81   0.81  0.84*   0.87   1.02   1.04   1.00   0.97   0.93
Decision Tree - DI      0.85   1.07   1.15   1.15   1.17   0.96   0.96   1.03   0.95   0.90
SVR (linear) - DI       0.81   1.04   1.09   1.12   1.22   1.06   1.06   0.98   1.01   1.02
SVR (polynomial) - DI   1.06   1.20   1.26   1.18   1.11   1.67   1.87   0.98   0.83   0.89
SVR (rbf) - DI          0.71   0.85   0.90   0.91   0.93   1.07   1.21   1.17   1.04   0.97
SVR (sigmoid) - DI      0.79   1.00   1.04   1.20   1.24   1.10   1.15   1.04   1.25   1.17
Ensemble machine learning models using dimension reduction
Random Forest - DI      0.85   1.07   1.14   1.14   1.15   0.96   0.96   1.03   0.95   0.89
XGBoost - DI            0.94   0.94   0.96   1.01   1.06   1.10   1.02   0.99   0.93   0.91
AdaBoost - DI           0.68   0.78   0.93   0.94   1.04   0.98   0.94   0.95   0.86  0.84*
Gradient Boost - DI     0.87   1.11   1.14   1.19   1.13   1.05   1.01   1.06   0.94   0.87

41










    Table 11: Employment: relative RMSPE (sample period: 1960m1-2023m12)

    Full out-of-sample        NBER recession periods
Model                  h=1    h=3    h=6    h=9    h=12   h=1    h=3     h=6      h=9      h=12
Baseline Model
ARD (RMSPE)           0.694  0.091  0.068  0.069  0.031  0.204  0.077   0.044    0.036    0.032
Individual machine learning models
kNN (uniform)          0.11   0.50   0.45   0.36   0.73   1.00   1.04    1.08     1.05     1.03
kNN (inverse)          0.11   0.50   0.45   0.36   0.73   1.00   1.04    1.08     1.05     1.02
Decision Tree          0.16   0.82   0.64   0.46   0.80   1.01   1.01    1.12     0.99     0.93
SVR (linear)           0.15   0.83   0.61   0.51   0.88   1.00   1.01    0.97   0.86***   0.76***
SVR (polynomial)       0.12   0.50   0.50   0.41   0.83   1.01   1.09    1.13     1.15     1.12
SVR (rbf)              0.11   0.47   0.42   0.35   0.72   1.01   1.04    1.05     0.98     0.90
SVR (sigmoid)          0.12   0.50   0.43   0.35   0.74   1.00   1.00    0.99   0.88***   0.79***
Ensemble machine learning models
Random Forest          0.13   0.67   0.50   0.40   0.76   1.00   1.00    1.05     0.95    0.88*
XGBoost                0.17   0.82   0.66   0.37   0.76   1.00   1.00    1.03     0.96     0.89
AdaBoost               0.11   0.78   0.52   0.39   0.72   1.00   1.00    1.05     0.97    0.88*
Gradient Boost         0.14   0.84   0.58   0.43   0.82   1.00   1.01    1.11     0.98     0.92
Individual machine learning models using dimension reduction
kNN (uniform) - DI     0.11   0.50   0.43   0.34   0.68   1.00   1.01    1.01     0.94    0.89*
kNN (inverse) - DI     0.11   0.50   0.44   0.34   0.68   1.00   1.01    1.01     0.94    0.89*
Decision Tree - DI     0.13   0.70   0.52   0.51   0.96   1.00   1.01    1.05     1.01     0.98
SVR (linear) - DI      0.12   0.50   0.44   0.38   0.81   1.00  0.98**  0.95**  0.87*** 0.76***
SVR (polynomial) - DI  0.14   0.72   0.89   0.76   1.54   1.18   2.12    2.97     2.17     1.00
SVR (rbf) - DI         0.11   0.47   0.43   0.36   0.75   1.01   1.06    1.09     1.01     0.92
SVR (sigmoid) - DI     0.15   0.69   0.70   0.63   1.32   1.26   1.66    1.91     1.52     1.37
Ensemble machine learning models using dimension reduction
Random Forest - DI     0.13   0.63   0.52   0.51   0.98   1.00   1.01    1.05     1.00     0.98
XGBoost - DI           0.18   0.74   0.54   0.44   0.97   1.00   0.99    1.01     0.97     0.92
AdaBoost - DI          0.11   0.49   0.49   0.45   1.00  1.00*   0.99    1.01     0.98     0.90
Gradient Boost - DI    0.12   0.68   0.57   0.53   1.08   1.00   1.00    1.04     1.01     0.98

42










    Table 12: Real Personal Income: relative RMSPE (sample period: 1960m1-2023m12)

    Full out-of-sample        NBER recession periods
Model                  h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9     h=12
Baseline Model
ARD (RMSPE)           0.162  0.063  0.043  0.034  0.028  0.184  0.075   0.05  0.041   0.037
Individual machine learning models
kNN (uniform)          0.97   1.01   0.89   0.89   1.00   1.00   1.00   1.02   1.06    1.05
kNN (inverse)          0.97   1.01   0.89   0.89   1.00   1.00   1.00   1.02   1.06    1.05
Decision Tree          1.15   1.28   1.11   1.09   1.16   1.16   1.02   1.08   0.97    1.03
SVR (linear)           1.07   1.23   1.11   1.18   1.29   1.09   1.14   0.98  0.89**  0.88*
SVR (polynomial)       0.96   1.00   0.94   1.00   1.10   1.05   1.09   0.98   1.04    1.03
SVR (rbf)              0.95   0.99   0.86   0.87   0.97   1.00   0.97  0.93**  0.97    0.96
SVR (sigmoid)          0.95   0.97   0.90   0.97   1.11   1.00 0.93*   0.87*   0.86   0.86*
Ensemble machine learning models
Random Forest          1.04   1.07   0.98   0.95   1.12   1.05   0.98   0.91   0.91    0.97
XGBoost                1.41   1.34   0.95   0.90   1.09   1.04   1.02   0.98   0.94    0.93
AdaBoost               1.02   1.03   0.87   0.85   0.94   0.99   0.97  0.88*  0.90* 0.88***
Gradient Boost         1.41   1.20   1.03   1.10   1.37   1.06   1.05   0.98   1.01    1.02
Individual machine learning models using dimension reduction
kNN (uniform) - DI     0.96   1.00   0.88   0.89   1.00   1.00   1.02   1.02   1.06    1.05
kNN (inverse) - DI     0.96   1.00   0.89   0.89   1.00   1.01   1.02   1.02   1.05    1.05
Decision Tree - DI     1.17   1.09   0.96   0.97   1.25   0.98   0.97   1.01   0.90    0.97
SVR (linear) - DI      0.99   1.03   1.02   1.08   1.21   1.02   0.92   1.06   1.08    1.03
SVR (polynomial) - DI  1.05   1.46   1.56   1.67   2.11   1.19   1.65   1.52   1.12    1.00
SVR (rbf) - DI         0.98   1.01   0.89   0.90   1.02   0.99   0.98   0.98   1.04    1.06
SVR (sigmoid) - DI     1.06   1.11   1.09   1.03   1.18   1.00   0.96   1.04   1.04    0.99
Ensemble machine learning models using dimension reduction
Random Forest - DI     1.16   1.07   0.97   0.97   1.24   0.98   0.96   1.01   0.92    0.99
XGBoost - DI           1.18   1.11   0.86   0.91   1.11   1.07   0.96   1.00   0.96    0.98
AdaBoost - DI          1.01   0.96   0.85   0.88   1.10   0.97 0.93**  0.93*   0.92    0.92
Gradient Boost - DI    1.28   1.21   0.93   1.01   1.26   1.10   0.89   0.97   0.92    1.00

43










    Table 13: Unemployment Rate: relative RMSPE (sample period: 1960m1-2023m12)

                               Full out-of-sample        NBER recession periods
Model                  h=1     h=3    h=6    h=9    h=12   h=1     h=3     h=6     h=9      h=12
Baseline Model
ARD (RMSPE)           18.839  6.249  3.762  2.464  2.183  14.913  5.716   3.345   2.578     2.14
Individual machine learning models
kNN (uniform)          0.30    0.54   0.59   0.69   0.65   0.97    1.00    1.00    0.96   0.94***
kNN (inverse)          0.30    0.54   0.59   0.69   0.65   0.97    1.00    1.00    0.96   0.94***
Decision Tree          0.38    0.88   0.74   0.85   0.83   0.98    1.03    1.03    0.96    0.85**
SVR (linear)           0.40    0.90   0.82   1.03   0.85   0.97    0.99    0.96  0.87***  0.76***
SVR (polynomial)       0.32    0.54   0.63   0.75   0.72   0.98    1.04    1.03    1.04     1.04
SVR (rbf)              0.30    0.51   0.58   0.70   0.69   0.98    1.01    1.01    0.96    0.92**
SVR (sigmoid)          0.32    0.55   0.58   0.70   0.69   0.97   0.98*    0.98   0.89**  0.82***
Ensemble machine learning models
Random Forest          0.34    0.72   0.69   0.77   0.76   0.97    0.99    1.01   0.93*   0.84***
XGBoost                0.50    0.93   0.89   0.81   0.76   0.97   0.98*    0.99    0.93    0.86**
AdaBoost               0.29    0.79   0.68   0.72   0.77   0.97    0.99    1.01   0.92*   0.86***
Gradient Boost         0.36    0.91   0.74   0.85   0.84   0.96    1.02    1.03    0.95    0.87**
Individual machine learning models using dimension reduction
kNN (uniform) - DI     0.30    0.56   0.60   0.69   0.67   0.98    1.00    1.00    0.97     0.95
kNN (inverse) - DI     0.30    0.56   0.60   0.69   0.67   0.98    1.00    1.00    0.97     0.95
Decision Tree - DI     0.36    0.71   0.81   1.05   0.97   0.94    0.99    0.97    0.95     0.91
SVR (linear) - DI      0.36    0.61   0.65   0.84   0.82   0.96    0.99    0.98 0.92***   0.87***
SVR (polynomial) - DI  0.34    0.52   0.65   0.80   0.75   0.99    0.87    1.05   0.88*     0.90
SVR (rbf) - DI         0.30    0.52   0.59   0.72   0.72   0.98    1.03    1.03    1.00     0.97
SVR (sigmoid) - DI     0.35    0.61   0.60   0.84   0.86   1.02    1.21    1.10    1.04     1.01
Ensemble machine learning models using dimension reduction
Random Forest - DI     0.35    0.68   0.78   1.02   0.96   0.94    0.99    0.97    0.95     0.90
XGBoost - DI           0.51    0.79   0.81   0.95   1.00   0.99   0.96**  0.92**  0.88*    0.82**
AdaBoost - DI          0.29    0.50   0.71   0.98   0.93   0.97   0.97**   0.96   0.92*    0.86**
Gradient Boost - DI    0.37    0.72   0.82   1.08   1.01   1.02    0.99    0.96    0.96    0.89*

44










    Table 14: Real PCE: relative RMSPE (sample period: 1960m1-2023m12)

    Full out-of-sample                           NBER recession periods
Model                   h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9    h=12
Baseline Model
ARD (RMSPE)            0.276  0.064  0.048   0.03  0.043  0.202  0.108  0.061  0.049  0.043
Individual machine learning models
kNN (uniform)          0.37*   0.90   0.73   0.94   0.58   1.01   0.99   1.02   1.03   1.08
kNN (inverse)          0.37*   0.90   0.73   0.94   0.58   1.01   0.99   1.02   1.03   1.08
Decision Tree           0.56   1.35   0.94   1.21   0.75   1.06   0.99   0.96   0.96   0.96
SVR (linear)           0.49*   1.46   1.04   1.26   0.69   1.05   0.99   0.97  0.85**  0.91
SVR (polynomial)       0.38*   0.89   0.76   0.98   0.61   1.03   1.01   1.05   1.07   1.09
SVR (rbf)              0.36*   0.85   0.71   0.94   0.58   1.01  0.98*   0.99   1.00   1.02
SVR (sigmoid)          0.38*   0.92   0.72   0.94   0.58   1.02   0.97   0.97  0.88*  0.90*
Ensemble machine learning models
Random Forest          0.44*   1.11   0.82   1.04   0.62   0.95   0.97   0.93   0.91   0.94
XGBoost                0.46*   1.36   1.12   1.03   0.65   1.00   0.99   0.97   0.99   1.02
AdaBoost               0.39*   1.13   0.77   0.99   0.57   1.01  0.97*   0.95   0.96   0.97
Gradient Boost         0.46*   1.23   0.97   1.13   0.76   0.98   0.98   0.96   0.93   0.98
Individual machine learning models using dimension reduction
kNN (uniform) - DI     0.37*   0.94   0.75   0.95   0.58   0.99  0.98*   1.00   1.03   1.05
kNN (inverse) - DI     0.37*   0.94   0.75   0.95   0.57   0.99  0.98*   1.00   1.03   1.05
Decision Tree - DI      0.50   1.16   1.04   1.29   0.74   1.06   1.00   1.04   1.05   1.07
SVR (linear) - DI      0.46*   1.07   0.87   1.16   0.69   1.04   0.97   0.99   0.97   1.00
SVR (polynomial) - DI  0.41*   0.93   0.90   1.24   0.70   1.08   0.97   1.03   1.03   1.09
SVR (rbf) - DI         0.37*   0.87   0.72   0.94   0.59   1.01   1.00   1.02   1.03   1.05
SVR (sigmoid) - DI     0.40*   0.82   0.68   1.17   0.72   1.15   0.97   0.95   1.00   1.04
Ensemble machine learning models using dimension reduction
Random Forest - DI      0.50   1.18   1.01   1.25   0.74   1.06   1.01   1.02   1.05   1.06
XGBoost - DI           0.52*   1.25   0.86   1.16   0.68   1.07   1.00   0.99   1.01   1.05
AdaBoost - DI          0.36*   0.85   0.78   1.11   0.64   1.00  0.97*   0.95   0.96   1.01
Gradient Boost - DI    0.46*   1.33   1.07   1.48   0.84   1.06   1.05   0.97   1.04   1.08

Industrial Production  Employment
1  0.95  0.94  0.93  1
0.9
0.8  0.78  0.87 0.81 0.89 0.84  0.86  0.9  0.91   0.87 0.87 0.89  0.92
0.8  Pre-Pandemic Sample
Full Sample
Relative RMSPE (vs ARD)  0.7  0.68  Relative RMSPE (vs ARD)  0.7  0.68
0.6  0.6

0.5  0.5  0.47
0.42
0.4  0.4  0.34
0.3  0.3

0.2  0.2
0.1  0.1  0.11

0  0
1  3  6  9  12  1  3  6   9  12
Forecast Horizon (months)  Forecast Horizon (months)
Real Personal Income  Unemployment Rate
1  0.920.95  0.950.96  0.92  0.930.94  1  0.92
0.9  0.890.85  0.85  0.9  0.87  0.83  0.86  0.87
0.8  0.8

Relative RMSPE (vs ARD)  0.7  Relative RMSPE (vs ARD)  0.7  0.69  0.65
0.6  0.6  0.58
0.5  0.5  0.50

0.4  0.4

0.3  0.3  0.29

0.2  0.2

0.1  0.1

0  0
1  3  6  9  12  1  3  6   9  12
Forecast Horizon (months)  Forecast Horizon (months)
Real Personal Consumption Expenditure
1  0.97  0.92  0.93  0.960.94   1.00
0.9
0.82
Relative RMSPE (vs ARD)  0.8
0.7  0.68

0.6  0.57

0.5

0.4  0.36

0.3

0.2

0.1

0
1  3  6  9  12
Forecast Horizon (months)

Figure 1: Relative RMSPE Pre-Post Covid
45

46










                                                              Figure 2:  RMSPE over time for real variables
            0.02                 36-month rolling RMSPE (1973-2019)                       0.03                36-month rolling RMSPE (1973-2023)
INDPRO      0.01                                                                  INDPRO  0.02
                                                                                          0.01

               0                                                                             0
                  1975    1980   1985   1990   1995   2000   2005  2010   2015                  1975   1980   1985   1990   1995   2000   2005   2010   2015   2020

                 #10-3
                                                                                          0.03
               2
       PAYEMS  1                                                                  PAYEMS  0.02
                                                                                          0.01

               0                                                                             0
                  1975    1980   1985   1990   1995   2000   2005  2010   2015                  1975   1980   1985   1990   1995   2000   2005   2010   2015   2020

               6 #10-3                                                                    0.02

               4
       RPI     2                                                                  RPI     0.01

               0                                                                             0
                  1975    1980   1985   1990   1995   2000   2005  2010   2015                  1975   1980   1985   1990   1995   2000   2005   2010   2015   2020



                 0.2                                                                          2
    UNRATE       0.1                                                                  UNRATE  1

                   0                                                                          0
                      1975   1980   1985   1990   1995   2000   2005   2010   2015                 1975   1980   1985   1990   1995   2000   2005       2010   2015   2020

                   6 #10-3
        Real PCE   4                                                              Real PCE 0.02                                              ARD
                                                                                                                                             AdaBoost
                                                                                                                                             AdaBoost-DI
                   2                                                                       0.01                                              kNN(uniform)
                                                                                                                                             kNN(uniform)-DI
                                                                                                                                             SVR(linear)-DI

                   0                                                                          0
                      1975   1980   1985   1990   1995   2000   2005   2010   2015                 1975   1980   1985   1990   1995   2000   2005       2010   2015   2020
Notes:  The figure shows the 3-year moving average of the RMSPE of the real variables for selected models at h = 3. The left panel covers
the original sample between 1973-2019, while the right panel includes obersvations till 2023.

    47










                                                                Figure 3:     RMSPE over time for nominal variables
    8 #10-3                                36-month rolling RMSPE (1973-2019)                                    8 #10-3          36-month rolling RMSPE (1973-2023)
    CPI (All items) 642                                                                  CPI (All items)         642
                      0                                                                                          0
                           1975     1980   1985   1990   1995   2000   2005   2010   2015                           1975   1980   1985   1990   1995   2000   2005   2010   2015   2020

                   0.01                                                                                  0.01
    CPI less food 0.005                                                                  CPI less food   0.005
                      0                                                                                          0
                           1975     1980   1985   1990   1995   2000   2005   2010   2015                           1975   1980   1985   1990   1995   2000   2005   2010   2015   2020

                      6    #10-3                                                                                 6 #10-3

        PCEPI        42                                                                                  PCEPI   42

                      0                                                                                          0
                           1975     1980   1985   1990   1995   2000   2005   2010   2015                           1975   1980   1985   1990   1995   2000   2005   2010   2015   2020



    PPI - FCG   0.02                                                                 PPI - FCG   0.02
                0.01                                                                             0.01


  0  0
  1975  1980 1985 1990  1995 2000  2005   2010 2015  1975   1980  1985 1990 1995 2000 2005  2010   2015   2020

  0.1  0.1
  ARD
  AdaBoost
  PPI - CM  0.05  PPI - CM  0.05  AdaBoost-DI
  kNN(uniform)
  kNN(uniform)-DI
  SVR(linear)-DI


        0 0
        1975 1980 1985 1990 1995 2000 2005 2010 2015 1975 1980 1985 1990 1995 2000 2005 2010 2015 2020
Notes: The figure shows the 3-year moving average of the RMSPE of the nominal variables for selected models at h = 3. The left panel
covers the original sample between 1973-2019, while the right panel includes observations till 2023.

48










                        Figure 4: RMSPE over time for financial variables
        0.06        36-month rolling RMSPE (1973-2019)    0.06  36-month rolling RMSPE (1973-2023)
S&P 500 0.04            S&P 500                           0.04
        0.02                                              0.02

           0                                                 0
    1975     1980   1985 1990 1995 2000 2005 2010     2015      1975 1980 1985 1990 1995 2000 2005 2010 2015 2020

           1                                                 1


GS1 0.5             GS1 0.5


                      0                                                                                         0
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

                    0.6                                                                                       0.6

     GS10           0.4                                                                    GS10               0.4
                    0.2                                                                                       0.2

                      0                                                                                         0
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

 US/UK Forex Rate  0.04                                                                    US/UK Forex Rate  0.04
                   0.03                                                                                      0.03

                   0.02                                                                                      0.02

                   0.01                                                                                      0.01

                      0                                                                                         0
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

 CA/US Forex Rate  0.03                                                                    CA/US Forex Rate  0.03         RW
                                                                                                                          AdaBoost
                   0.02                                                                                      0.02         AdaBoost-DI
                                                                                                                          kNN(uniform)
                                                                                                                          kNN(uniform)-DI
                   0.01                                                                                      0.01         SVR(linear)-DI

                      0                                                                                         0
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020
Notes:     The figure shows the 3-year moving average of the RMSPE of the real variables for selected models at h = 3. The left panel covers
the original sample between 1973-2019, while the right panel includes observations till 2023.

49










    Figure 5:   Cumulative errors over time for real variables

    Cumulative errors (1973-2019)                           Cumulative errors (1973-2023)

         0.1
INDPRO -0.10        INDPRO                              -0.20
        -0.2
        -0.3                                            -0.4
    1975     1980 1985 1990 1995 2000 2005 2010 2015        1975 1980 1985 1990 1995 2000 2005 2010 2015 2020


           0                                               0
PAYEMS -0.05        PAYEMS                              -0.1
        -0.1                                            -0.2
       -0.15                                            -0.3
        -0.2                                            -0.4
    1975     1980 1985 1990 1995 2000 2005 2010 2015        1975 1980 1985 1990 1995 2000 2005 2010 2015 2020



               0                                                                                  0
       RPI  -0.1                                                                RPI            -0.1
            -0.2                                                                               -0.2
            -0.3                                                                               -0.3
            -0.4                                                                               -0.4
                    1975   1980   1985   1990   1995   2000   2005   2010   2015                     1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

               5                                                                                 20
       UNRATE  0                                                                    UNRATE       10
                                                                                                  0
              -5                                                                                -10

             -10                                                                                -20
                    1975   1980   1985   1990   1995   2000   2005   2010   2015                     1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

                                                                                                0.1
               0
Real PCE   -0.05                                                                    Real PCE  -0.10         ARD
            -0.1                                                                                            AdaBoost
                                                                                                            AdaBoost-DI
           -0.15                                                                               -0.2         kNN(uniform)
                                                                                                            kNN(uniform)-DI
            -0.2                                                                               -0.3         SVR(linear)-DI
                    1975   1980   1985   1990   1995   2000   2005   2010   2015                     1975   1980          1985   1990   1995   2000   2005   2010   2015   2020
Notes:   The figure shows the cumulative errors of the real variables for selected models at h = 3. The left panel covers the original sample
between 1973-2019, while the right panel includes obersvations till 2023.

50










    Figure 6:                           Cumulative errors over time for nominal variables
     0.05    Cumulative errors (1973-2019)             0.05  Cumulative errors (1973-2023)

        0                                                 0
CPI -0.05                                   CPI       -0.05

     -0.1                                              -0.1
    1975     1980 1985 1990 1995 2000   2005  2010 2015      1975 1980 1985 1990 1995    2000 2005 2010 2015 2020


 CPI less food    0.050                                                                    CPI less food  0.050
                  -0.05                                                                                   -0.05
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980   1985   1990   1995   2000   2005   2010   2015     2020

                   0.04                                                                                    0.04

                   0.02                                                                                    0.02
 PCEPI           -0.020                                                                    PCEPI         -0.020

                  -0.04                                                                                   -0.04
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980   1985   1990   1995   2000   2005   2010   2015     2020

                    0.1                                                                                     0.2
       PPI - FCG  -0.10                                                                     PPI - FCG      0.10
                                                                                                           -0.1

                   -0.2                                                                                    -0.2
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980   1985   1990   1995   2000   2005   2010   2015     2020

                      1                                                                                       1
                    0.5                                                                                     0.5
       PPI - CM   -0.50                                                                     PPI - CM      -0.50                                                                  ARD
                                                                                                                                                                                 AdaBoost
                                                                                                                                                                                 AdaBoost-DI
                     -1                                                                                      -1                                                                  kNN(uniform)
                                                                                                                                                                                 kNN(uniform)-DI
                   -1.5                                                                                    -1.5                                                                  SVR(linear)-DI
                           1975   1980   1985   1990   1995   2000   2005   2010   2015                            1975   1980   1985   1990   1995   2000   2005   2010   2015     2020
Notes:     The figure shows the cumulative errors of the nominal variables for selected models at h = 3. The left panel covers the original
sample between 1973-2019, while the right panel includes observations till 2023.

51










                                                       Figure 7:     Cumulative errors over time for financial variables
                    1.5                      Cumulative errors (1973-2019)                                      1.5                           Cumulative errors (1973-2023)
                      1                                                                                           1
 S&P 500           0.50                                                                  S&P 500               0.50
                   -0.5                                                                                        -0.5
                     -1                                                                                          -1
                           1975   1980   1985   1990   1995   2000   2005     2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

                     20                                                                                          20

                     10                                                                                          10
 GS1                  0                                                                      GS1                  0
                    -10                                                                                         -10

                    -20                                                                                         -20
                           1975   1980   1985   1990   1995   2000   2005     2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

                     10                                                                                          10

                      5                                                                                           5
 GS10                 0                                                                      GS10                 0
                     -5                                                                                          -5

                    -10                                                                                         -10
                           1975   1980   1985   1990   1995   2000   2005     2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

 US/UK Forex Rate  0.50                                                                      US/UK Forex Rate  0.50
                   -0.5                                                                                        -0.5

                     -1                                                                                          -1
                           1975   1980   1985   1990   1995   2000   2005     2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020

 CA/US Forex Rate  0.2                                                                       CA/US Forex Rate  0.5
                      0                                                                                           0         ARD
                   -0.2                                                                                                     AdaBoost
                   -0.4                                                                                        -0.5         AdaBoost-DI
                                                                                                                            kNN(uniform)
                   -0.6                                                                                                     kNN(uniform)-DI
                                                                                                                 -1         SVR(linear)-DI
                           1975   1980   1985   1990   1995   2000   2005     2010   2015                            1975   1980          1985   1990   1995   2000   2005   2010   2015   2020
Notes:     The figure shows the cumulative errors of the real variables for selected models at h = 3. The left panel covers the original sample
between 1973-2019, while the right panel includes observations till 2023.

52










    Table 15: Real Personal Income: relative RMSPE (sample period: 1960m1-2019m12)

                              Pre-Pandemic Sample                NBER recession periods
Model                h=1      h=3     h=6    h=9    h=12   h=1    h=3     h=6    h=9     h=12
Baseline Model
ARD (RMSPE)          0.075   0.038   0.027  0.024  0.022  0.105  0.061   0.045  0.039   0.037
Individual machine learning models
kNN (uniform)         0.95    0.99   0.94*   0.95   0.97   0.96   0.97   0.98*   1.02    1.02
kNN (inverse)         0.96    0.99   0.94*   0.95   0.97   0.96   0.97   0.98**  1.02    1.02
Decision Tree         1.29    1.33    1.21   1.18   1.22   1.25   1.25    0.99   0.92    1.13
SVR (linear)          1.15    1.20    1.19   1.20   1.18   1.07   1.18    0.93   0.80   0.84*
SVR (polynomial)      0.96    1.01    1.09   1.18   1.16   1.07   1.11    0.99   1.05    1.04
SVR (rbf)            0.92**   0.95   0.94*   0.96   0.98   0.91   0.94   0.92**  0.95    0.96
SVR (sigmoid)        0.94*    0.97    0.96   1.04   1.08   0.90  0.88**  0.80**  0.83   0.84*
Ensemble machine learning models
Random Forest         0.98    1.01    1.00   1.01   1.04   0.99   0.98    0.91   0.90    1.00
XGBoost               1.07    1.06    0.97   0.99   1.00   1.12   1.04    0.94   0.92    0.93
AdaBoost             0.92**  0.94*  0.90***  0.92   0.93   0.92   0.95   0.85**  0.87  0.88***
Gradient Boost        1.18    1.21    1.18   1.16   1.22   1.13   1.08    0.98   0.91    1.11
Individual machine learning models using dimension reduction
kNN (uniform)-DI      0.96    0.98    0.94  0.92*  0.90*   0.93   0.94    0.92   0.92    0.88
kNN (inverse)-DI      0.96    0.98    0.94  0.92*  0.90*   0.93   0.94    0.92   0.92    0.88
Decision Tree-DI      1.12    1.18    1.17   1.11   1.12   1.09   1.06    0.97   0.86    0.87
SVR (linear)-DI       0.96    1.09    1.16   1.21   1.24   0.96   1.00    1.02   1.05    1.02
SVR (polynomial)-DI   1.46    1.85    1.90   2.09   2.09   1.35   1.48    1.25   1.16    1.00
SVR (rbf)-DI         0.93*    0.99    1.00   1.02   1.05   0.90   0.97    0.97   1.04    1.07
SVR (sigmoid)-DI      1.20    1.21    1.17   1.16   1.19   1.26   1.08    0.97   1.02    1.00
Ensemble machine learning models using dimension reduction
Random Forest-DI      1.13    1.20    1.17   1.11   1.11   1.08   1.06    0.98   0.84    0.88
XGBoost-DI            1.09    1.04    1.00   1.03   1.01   1.14   1.04    0.97   0.96    0.93
AdaBoost-DI           0.98   0.95*    0.95   0.94   0.95    0.86* 0.94   0.91**  0.88    0.88
Gradient Boost-DI     1.14    1.15    1.10   1.08   1.14   1.18   1.07    0.98   0.88    0.88

53










    Table 16: Unemployment Rate: relative RMSPE (sample period: 1960m1-2019m12)

                             Pre-Pandemic Sample                    NBER recession periods
Model                h=1     h=3     h=6     h=9      h=12    h=1     h=3    h=6      h=9 h=12
Baseline Model
ARD (RMSPE)         2.011    1.33   1.216   1.158    1.126   2.711   2.204  1.999    1.846    1.672
Individual machine learning models
kNN (uniform)        0.97    0.92    0.89    0.92     0.93    1.03    1.06   1.08     1.09     1.05
kNN (inverse)        0.97    0.92    0.89    0.92     0.93    1.02    1.06   1.08     1.09     1.04
Decision Tree        1.29    1.18    1.24    1.11     1.07    1.15    1.04   1.03     0.91 0.74*
SVR (linear)         1.11    1.05    0.95    0.94     0.98    1.01    0.91  0.83*   0.71***  0.56***
SVR (polynomial)     1.02    1.14    1.18    1.12     1.07    1.15    1.23   1.11     1.10     1.07
SVR (rbf)            0.98    0.95    0.92    0.93     0.94    1.05    1.11   1.03     0.93    0.86**
SVR (sigmoid)        1.00    0.94    0.93    0.91     0.91    0.95   0.89*   0.91    0.78**  0.68***
Ensemble machine learning models
Random Forest       0.95**   0.96    1.01    0.97     0.95  0.84***   0.92   1.02    0.83**   0.71**
XGBoost              1.01    0.92    0.98    0.92     0.92   0.85**  0.88**  0.99     0.87    0.75**
AdaBoost           0.93***  0.89**   0.92    0.90     0.90   0.90**   0.93   1.02     0.86    0.74**
Gradient Boost       1.25    1.17    1.22    1.07     1.05    1.07    1.02   1.02     0.88    0.77**
Individual machine learning models using dimension reduction
kNN (uniform)-DI    0.96**  0.87**  0.78**  0.76**  0.75***  0.91*   0.89*  0.84**   0.76**  0.70***
kNN (inverse)-DI    0.96**  0.87**  0.78**  0.76**  0.75***  0.91*   0.89*  0.84**   0.76**  0.70***
Decision Tree-DI     1.24    1.12    1.09    1.05     1.14    1.14    1.01   1.05     0.88     0.87
SVR (linear)-DI      1.18    1.07    0.98    1.02     1.06    1.13    0.98  0.85**  0.80***  0.71***
SVR (polynomial)-DI  1.18    0.99    1.10    1.10     1.16    1.62    1.16   1.29    0.73*     0.81
SVR (rbf)-DI         1.05    1.01    0.97    0.97     0.98    1.09    1.18   1.09     0.99     0.94
SVR (sigmoid)-DI     1.19    1.06    1.07    1.03     1.10    1.31    1.05   1.11     1.03     0.99
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.23    1.11    1.07    1.05     1.14    1.15    0.98   1.05     0.85     0.88
XGBoost-DI           1.08    0.95    0.89    0.96     1.03    1.01   0.78**  0.82    0.73**   0.68**
AdaBoost-DI         0.96**  0.87**  0.83*    0.89     0.94    0.93   0.83**  0.87    0.79**   0.75**
Gradient Boost-DI    1.24    1.14    1.04    1.03     1.10    1.25    1.03   0.99     0.86     0.85

54










Table 17: Real Personal Consumption Expenditures: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample        NBER recession periods
Model                h=1    h=3     h=6    h=9 h=12      h=1     h=3       h=6    h=9    h=12
Baseline Model
ARD (RMSPE)          0.06   0.03   0.022   0.02  0.019  0.086     0.053    0.04  0.038  0.037
Individual machine learning models
kNN (uniform)        1.04   1.01    1.05   1.06   1.06   0.99      1.00    1.14   1.13   1.10
kNN (inverse)        1.04   1.01    1.05   1.06   1.05   0.99      1.00    1.15   1.13   1.09
Decision Tree        1.32   1.21    1.17   1.12   1.14   1.12      0.94    0.85   0.89   0.96
SVR (linear)         1.17   1.08    1.13   1.11   1.15   1.13      0.93    0.86  0.77**  0.88
SVR (polynomial)     1.00   1.03    1.14   1.12   1.13   1.02      1.04    1.10   1.11   1.11
SVR (rbf)           0.97** 0.92***  0.95   0.99   1.01  0.92**    0.90*    0.96   1.00   1.03
SVR (sigmoid)        1.05   0.96    0.99   1.01   1.04   1.01     0.81*   0.88*  0.80*   0.86
Ensemble machine learning models
Random Forest        1.01   0.96    0.96   0.95   0.99   0.92     0.82**  0.81*   0.85   0.95
XGBoost              1.01   1.02    0.95   1.01   1.03   0.93      0.91    0.91   0.96   1.01
AdaBoost             0.98 0.95*     0.93   0.99   1.00  0.92*    0.85***  0.88*   0.91   0.94
Gradient Boost       1.25   1.19    1.14   1.11   1.13   1.06      0.88    0.83   0.88   0.97
Individual machine learning models using dimension reduction
kNN (uniform)-DI     0.99  0.93**  0.91*  0.89*   0.90   0.93     0.87**   0.90   0.89   0.95
kNN (inverse)-DI     1.00  0.93**  0.92*  0.89*   0.90   0.93     0.86**   0.90   0.90   0.95
Decision Tree-DI     1.26   1.21    1.16   1.21   1.20   1.19      1.08    0.97   1.00   1.10
SVR (linear)-DI      1.11   1.09    1.16   1.23   1.23   1.12      0.94    0.99   0.94   1.00
SVR (polynomial)-DI  1.24   1.23    1.16   1.18   1.25   1.42      1.25    1.02   1.03   1.10
SVR (rbf)-DI         1.02   1.01    1.00   1.01   1.03   0.95      1.02    1.05   1.05   1.06
SVR (sigmoid)-DI     1.08   1.02    1.10   1.05   1.09   0.94      0.87    1.05   0.94   1.03
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.28   1.21    1.16   1.20   1.20   1.22      1.09    0.94   0.98   1.11
XGBoost-DI           1.10   1.03    0.98   1.03   1.06   1.15      1.05    0.97   1.01   1.08
AdaBoost-DI          0.97  0.94**   0.94   0.98   1.01  0.91**   0.90*     0.88   0.91   1.01
Gradient Boost-DI    1.20   1.21    1.15   1.16   1.18   1.09      1.13    0.94   1.00   1.10

55










Table 18: CPI (All items less food) inflation: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                      NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9    h=12
Baseline Model
ARD (RMSPE)         0.034   0.03  0.026  0.025  0.024  0.057  0.055  0.043  0.039  0.038
Individual machine learning models
kNN (uniform)        1.10   1.24   1.39   1.47   1.50   0.93   1.10   1.48   1.19   1.16
kNN (inverse)        1.10   1.24   1.39   1.46   1.49   0.93   1.11   1.47   1.18   1.16
Decision Tree        1.44   1.64   1.65   1.60   1.57   1.16   1.63   1.47   1.25   1.12
SVR (linear)         1.24   1.37   1.54   1.54   1.51   1.07   1.07   1.25   1.19   1.17
SVR (polynomial)     1.06   1.29   1.43   1.44   1.48   1.00   1.27   1.35   1.10   1.12
SVR (rbf)            1.06   1.19   1.33   1.40   1.42   0.95   1.08   1.41   1.15   1.14
SVR (sigmoid)        1.13   1.28   1.42   1.51   1.48   1.05   1.08   1.37   1.17   1.14
Ensemble machine learning models
Random Forest        1.11   1.27   1.33   1.26   1.30   1.03   1.34   1.36   1.01   1.02
XGBoost              1.14   1.22   1.25   1.28   1.32   1.05   1.18   1.26   0.97   1.04
AdaBoost             1.06   1.19   1.24   1.26   1.29   0.96   1.18   1.33   1.01   0.97
Gradient Boost       1.34   1.50   1.53   1.44   1.50   1.20   1.50   1.49   1.18   1.14
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.08   1.23   1.37   1.40   1.41   0.96   1.13   1.44   1.07   1.01
kNN (inverse)-DI     1.09   1.23   1.37   1.40   1.41   0.96   1.14   1.44   1.07   1.01
Decision Tree-DI     1.27   1.33   1.46   1.58   1.55   0.92   1.18   1.44   1.15   1.16
SVR (linear)-DI      1.35   1.59   1.99   2.06   2.03   1.26   1.48   2.12   1.72   1.54
SVR (polynomial)-DI  1.60   1.73   2.08   2.27   2.37   1.84   1.77   2.39   2.03   1.76
SVR (rbf)-DI         1.11   1.25   1.46   1.56   1.57   0.98   1.11   1.49   1.21   1.18
SVR (sigmoid)-DI     1.24   1.48   1.82   1.95   1.97   1.06   1.44   2.01   1.76   1.45
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.29   1.31   1.46   1.59   1.55   0.95   1.14   1.43   1.18   1.16
XGBoost-DI           1.11   1.14   1.29   1.32   1.34   0.98   1.04   1.25   1.06   0.97
AdaBoost-DI          1.02   1.11   1.19   1.23   1.26   0.89   0.98   1.22   1.00   1.00
Gradient Boost-DI    1.20   1.28   1.36   1.55   1.52   0.99   1.14   1.36   1.22   1.14

56










    Table 19: PCEPI inflation: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                      NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9    h=12
Baseline Model
ARD (RMSPE)         0.022  0.019  0.017  0.016  0.016  0.036  0.034  0.028  0.027  0.026
Individual machine learning models
kNN (uniform)        1.13   1.29   1.43   1.46   1.47   1.02   1.15   1.49   1.25   1.25
kNN (inverse)        1.13   1.30   1.44   1.46   1.47   1.03   1.16   1.49   1.25   1.25
Decision Tree        1.46   1.60   1.64   1.61   1.68   1.26   1.50   1.36   1.16   1.34
SVR (linear)         1.26   1.40   1.54   1.52   1.53   1.02   1.07   1.21   1.22   1.28
SVR (polynomial)     1.07   1.30   1.43   1.40   1.42   1.00   1.24   1.32   1.15   1.20
SVR (rbf)            1.07   1.22   1.35   1.38   1.39   0.95   1.08   1.37   1.15   1.19
SVR (sigmoid)        1.12   1.28   1.43   1.45   1.44   0.98   1.06   1.27   1.14   1.18
Ensemble machine learning models
Random Forest        1.13   1.27   1.35   1.29   1.29   1.08   1.27   1.27   1.01   1.05
XGBoost              1.17   1.23   1.26   1.27   1.32   1.03   1.12   1.22   1.02   1.07
AdaBoost             1.06   1.20   1.24   1.24   1.26   0.96   1.15   1.24   1.02   1.07
Gradient Boost       1.40   1.55   1.55   1.57   1.56   1.13   1.47   1.28   1.12   1.33
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.12   1.28   1.40   1.38   1.38   1.01   1.14   1.39   1.09   1.10
kNN (inverse)-DI     1.12   1.28   1.40   1.38   1.38   1.02   1.15   1.39   1.09   1.10
Decision Tree-DI     1.35   1.34   1.57   1.68   1.59   1.10   1.06   1.43   1.36   1.27
SVR (linear)-DI      1.42   1.62   1.98   1.97   1.91   1.37   1.42   1.95   1.66   1.60
SVR (polynomial)-DI  1.44   1.64   1.95   2.05   2.04   1.35   1.34   1.91   2.11   2.45
SVR (rbf)-DI         1.11   1.26   1.47   1.52   1.53   0.96   1.09   1.43   1.21   1.20
SVR (sigmoid)-DI     1.27   1.53   1.79   1.82   1.79   1.21   1.49   1.80   1.34   1.42
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.36   1.33   1.58   1.66   1.60   1.11   1.06   1.44   1.32   1.27
XGBoost-DI           1.09   1.16   1.29   1.35   1.34   0.98   1.04   1.29   1.15   1.11
AdaBoost-DI          1.04   1.11   1.22   1.24   1.28   0.93   0.97   1.20   1.03   1.11
Gradient Boost-DI    1.26   1.25   1.49   1.60   1.63   1.08   0.93   1.43   1.23   1.33

57










Table 20: Producer Price Index (Finished Consumer Goods) inflation: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                      NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9    h=12
Baseline Model
ARD (RMSPE)         0.093   0.07   0.06  0.055  0.053  0.148  0.118  0.092   0.08  0.065
Individual machine learning models
kNN (uniform)        1.20   1.41   1.60   1.70   1.75   1.01   1.25   1.71   1.66   1.92
kNN (inverse)        1.20   1.41   1.60   1.70   1.75   1.01   1.25   1.71   1.65   1.92
Decision Tree        1.47   1.51   1.66   1.87   1.76   1.21   1.26   1.58   1.45   1.55
SVR (linear)         1.36   1.58   1.78   1.84   1.84   1.17   1.25   1.59   1.71   1.97
SVR (polynomial)     1.15   1.42   1.60   1.64   1.69   1.03   1.32   1.60   1.59   1.82
SVR (rbf)            1.16   1.35   1.53   1.62   1.67   1.01   1.21   1.65   1.59   1.83
SVR (sigmoid)        1.22   1.44   1.65   1.76   1.76   1.04   1.22   1.60   1.64   1.88
Ensemble machine learning models
Random Forest        1.16   1.26   1.40   1.48   1.45   1.03   1.21   1.47   1.33   1.49
XGBoost              1.21   1.31   1.42   1.48   1.49   1.02   1.23   1.47   1.38   1.60
AdaBoost             1.12   1.23   1.37   1.43   1.46   0.97   1.13   1.51   1.40   1.57
Gradient Boost       1.42   1.43   1.66   1.76   1.78   1.12   1.23   1.61   1.37   1.61
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.19   1.40   1.60   1.65   1.68   1.00   1.26   1.69   1.54   1.71
kNN (inverse)-DI     1.20   1.40   1.60   1.65   1.68   1.01   1.26   1.69   1.54   1.71
Decision Tree-DI     1.41   1.54   1.71   1.78   1.82   1.15   1.30   1.68   1.47   1.65
SVR (linear)-DI      1.53   1.68   2.11   2.20   2.20   1.37   1.48   2.05   1.98   2.31
SVR (polynomial)-DI  2.04   2.39   3.28   3.47   3.22   1.43   1.84   2.77   4.21   5.70
SVR (rbf)-DI         1.22   1.40   1.66   1.78   1.82   0.98   1.19   1.69   1.65   1.92
SVR (sigmoid)-DI     1.45   1.76   2.06   2.22   2.11   1.27   1.66   2.18   2.31   2.14
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.42   1.53   1.73   1.78   1.83   1.14   1.31   1.65   1.46   1.63
XGBoost-DI           1.25   1.35   1.41   1.50   1.50   1.15   1.34   1.40   1.41   1.50
AdaBoost-DI          1.10   1.23   1.34   1.42   1.44   0.95   1.12   1.45   1.35   1.56
Gradient Boost-DI    1.32   1.50   1.58   1.69   1.72   1.23   1.29   1.54   1.48   1.55

58










Table 21: Producer Price Index (Crude Materials) inflation: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                     NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1    h=3   h=6    h=9    h=12
Baseline Model
ARD (RMSPE)         0.457  0.321  0.285  0.268  0.258  0.681  0.552  0.51  0.458  0.362
Individual machine learning models
kNN (uniform)        1.20   1.44   1.61   1.70   1.76   0.99   1.27  1.53   1.52   1.64
kNN (inverse)        1.20   1.44   1.61   1.71   1.76   0.99   1.27  1.54   1.52   1.63
Decision Tree        1.58   1.67   2.01   2.03   1.91   1.17   1.35  1.90   1.87   1.58
SVR (linear)         1.47   1.75   2.00   2.07   2.05   1.26   1.41  1.49   1.52   1.61
SVR (polynomial)     1.17   1.42   1.64   1.72   1.74   0.96   1.29  1.47   1.49   1.56
SVR (rbf)            1.16   1.39   1.55   1.63   1.68   0.91   1.21  1.50   1.55   1.66
SVR (sigmoid)        1.25   1.53   1.70   1.80   1.83   0.98   1.31  1.49   1.57   1.67
Ensemble machine learning models
Random Forest        1.22   1.30   1.49   1.57   1.56   0.97   1.18  1.49   1.39   1.42
XGBoost              1.34   1.42   1.56   1.69   1.67   1.19   1.46  1.52   1.43   1.58
AdaBoost             1.14   1.30   1.43   1.48   1.52   0.91   1.22  1.41   1.40   1.45
Gradient Boost       1.56   1.59   1.97   2.06   2.05   1.24   1.35  1.98   2.27   2.48
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.20   1.44   1.60   1.61   1.61   0.91   1.26  1.48   1.46   1.37
kNN (inverse)-DI     1.20   1.44   1.60   1.61   1.61   0.92   1.26  1.48   1.46   1.36
Decision Tree-DI     1.48   1.57   1.80   1.76   1.78   1.46   1.38  1.51   1.50   1.63
SVR (linear)-DI      1.37   1.64   2.00   2.14   2.19   1.19   1.36  1.74   1.82   1.99
SVR (polynomial)-DI  3.24   4.00   4.10   4.48   4.75   1.41   1.91  2.87   6.01   8.17
SVR (rbf)-DI         1.20   1.43   1.65   1.76   1.82   0.88   1.22  1.56   1.63   1.80
SVR (sigmoid)-DI     1.57   1.84   2.11   2.14   2.28   1.18   1.56  1.82   1.91   2.34
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.48   1.59   1.79   1.77   1.80   1.40   1.39  1.50   1.51   1.66
XGBoost-DI           1.60   1.69   1.69   1.83   1.68   1.85   1.71  1.51   1.69   1.62
AdaBoost-DI          1.15   1.27   1.38   1.45   1.48   0.90   1.17  1.40   1.43   1.49
Gradient Boost-DI    1.57   1.61   1.69   1.80   1.78   1.37   1.47  1.47   1.46   1.69

59










    Table 22: 1-Year Treasury Rate: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample        NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1     h=3     h=6      h=9    h=12
Baseline Model
RW (RMSPE)          5.347  3.687  2.529  1.944  1.719  10.852  7.337   4.112    2.952  2.553
Individual machine learning models
kNN (uniform)        0.98   1.00   0.99   1.04   1.06  0.94*    0.93   0.86**    0.96   0.99
kNN (inverse)        0.98   1.00   0.99   1.05   1.06   0.94    0.93   0.87**    0.97   0.99
Decision Tree        1.32   1.65   1.46   1.40   1.62   1.19    1.56    1.50     1.43   1.49
SVR (linear)         1.15   1.20   1.21   1.36   1.60   1.10    1.02    1.00     1.21   1.05
SVR (polynomial)     1.05   1.18   1.32   1.37   1.38   1.05    1.10    0.99     1.02   1.06
SVR (rbf)            1.01   1.02   1.02   1.06   1.13   1.00    1.00    0.98     0.94   0.90
SVR (sigmoid)        1.06   1.09   1.11   1.22   1.37   1.01    0.95    1.02     1.04   0.94
Ensemble machine learning models
Random Forest        1.08   1.21   1.18   1.26   1.31   1.10    1.15    1.21     1.33   1.20
XGBoost              1.12   1.20   1.26   1.29   1.46   1.20    1.22    1.36     1.40   1.39
AdaBoost             0.98   1.09   1.06   1.20   1.35   1.00    1.01    0.97     1.28   1.26
Gradient Boost       1.33   1.47   1.34   1.37   1.55   1.26    1.30    1.38     1.43   1.43
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.00   1.02   1.01   1.04   1.10   0.98    0.98  0.92***   0.92**  0.95
kNN (inverse)-DI     1.00   1.02   1.01   1.04   1.10   0.98    0.98  0.92***  0.91***  0.95
Decision Tree-DI     1.21   1.33   1.30   1.32   1.41   1.01    1.17    1.18     1.25   1.22
SVR (linear)-DI      1.14   1.23   1.50   1.51   1.68   1.05    1.05    1.42     1.25   1.24
SVR (polynomial)-DI  1.83   2.77   2.72   1.86   1.89   1.74    3.66    3.10     1.33   1.14
SVR (rbf)-DI         1.01   1.11   1.17   1.15   1.20   1.00    1.06    1.09     1.04   1.05
SVR (sigmoid)-DI     1.18   1.25   1.40   1.45   1.47   1.07    1.06    0.97     1.16   1.09
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.22   1.33   1.28   1.33   1.40   1.00    1.17    1.18     1.25   1.23
XGBoost-DI           1.17   1.16   1.12   1.13   1.26   1.18    1.22    1.18     1.15   1.21
AdaBoost-DI          1.02   1.04   1.02   1.08   1.14   0.99    1.01 0.94*       1.03   1.00
Gradient Boost-DI    1.18   1.28   1.24   1.27   1.35   1.14    1.26    1.09     1.25   1.23

60










Table 23: 10-Year Treasury Rate: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                      NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9   h=12
Baseline Model
RW (RMSPE)          3.557  2.389  1.694  1.362  1.206  6.076  3.911  2.187  1.682  1.46
Individual machine learning models
kNN (uniform)        1.01   1.04   1.07   1.11   1.12   0.97   0.98  0.96*   0.98  1.00
kNN (inverse)        1.01   1.04   1.07   1.11   1.12   0.97   0.98   0.96   0.98  1.00
Decision Tree        1.38   1.42   1.43   1.51   1.63   1.31   1.22   1.26   1.56  1.66
SVR (linear)         1.24   1.27   1.30   1.42   1.54   1.19   1.17   1.21   1.31  1.14
SVR (polynomial)     1.04   1.09   1.10   1.13   1.18   1.07   1.11   1.01   1.02  1.01
SVR (rbf)            1.03   1.05   1.08   1.14   1.18   1.03   1.04   1.06   1.03  0.98
SVR (sigmoid)        1.09   1.10   1.16   1.27   1.34   1.12   1.10   1.11   1.10  0.97
Ensemble machine learning models
Random Forest        1.05   1.15   1.17   1.27   1.38   1.08   1.12   1.14   1.41  1.35
XGBoost              1.14   1.24   1.21   1.31   1.42   1.25   1.35   1.21   1.48  1.45
AdaBoost             1.00   1.09   1.12   1.24   1.36   1.03   1.09   1.06   1.43  1.26
Gradient Boost       1.33   1.44   1.36   1.50   1.59   1.28   1.43   1.20   1.57  1.65
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.01   1.09   1.10   1.13   1.19   0.98   1.05   1.01   0.99  1.05
kNN (inverse)-DI     1.01   1.09   1.10   1.13   1.19   0.98   1.04   1.01   0.99  1.05
Decision Tree-DI     1.23   1.39   1.40   1.44   1.44   1.11   1.31   1.25   1.16  1.16
SVR (linear)-DI      1.15   1.26   1.42   1.43   1.56   1.00   1.06   1.57   1.23  1.13
SVR (polynomial)-DI  1.33   1.46   1.47   1.21   1.26   1.56   1.69   1.90   1.02  1.08
SVR (rbf)-DI         1.05   1.12   1.17   1.17   1.24   1.01   1.12   1.14   1.12  1.08
SVR (sigmoid)-DI     1.17   1.22   1.26   1.27   1.34   1.08   1.15   1.07   1.09  1.01
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.24   1.42   1.41   1.43   1.46   1.11   1.35   1.29   1.17  1.20
XGBoost-DI           1.11   1.23   1.22   1.24   1.29   1.14   1.28   1.27   1.16  1.20
AdaBoost-DI          1.04   1.08   1.12   1.15   1.24   1.11   1.05   1.05   1.02  1.15
Gradient Boost-DI    1.28   1.33   1.32   1.39   1.41   1.29   1.35   1.26   1.13  1.20

61










Table 24: US/UK Foreign Exchange Rate: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                     NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1    h=3   h=6    h=9    h=12
Baseline Model
RW (RMSPE)          0.281  0.197  0.149  0.122  0.106  0.328  0.257  0.21  0.178  0.153
Individual machine learning models
kNN (uniform)        1.05   1.06   1.09   1.10   1.09   1.03   1.07  1.09   1.07   1.00
kNN (inverse)        1.05   1.06   1.09   1.10   1.10   1.03   1.07  1.09   1.06   1.00
Decision Tree        1.37   1.43   1.48   1.38   1.35   1.31   1.33  1.20   1.10   1.25
SVR (linear)         1.30   1.34   1.36   1.40   1.35   1.22   1.32  1.23   1.17   0.99
SVR (polynomial)     1.02   1.10   1.12   1.09   1.10   1.07   1.21  1.04   1.11   1.07
SVR (rbf)            1.03   1.06   1.07   1.07   1.05   1.01   1.03  1.05   1.04   0.96
SVR (sigmoid)        1.13   1.14   1.16   1.16   1.16   1.18   1.13  1.12   0.99   0.88
Ensemble machine learning models
Random Forest        1.04   1.19   1.27   1.22   1.17   0.96   1.23  1.10   1.14   1.11
XGBoost              1.05   1.18   1.27   1.24   1.22   0.94   1.23  1.10   1.06   1.10
AdaBoost             1.01   1.11   1.21   1.22   1.21   0.98   1.14  1.07   1.14   1.14
Gradient Boost       1.34   1.38   1.43   1.34   1.31   1.27   1.31  1.17   1.10   1.23
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.05   1.10   1.11   1.10   1.08   1.00   1.04  0.99   0.96  0.88**
kNN (inverse)-DI     1.05   1.10   1.11   1.10   1.07   0.99   1.05  0.99   0.96  0.87**
Decision Tree-DI     1.29   1.39   1.35   1.40   1.37   1.23   1.34  1.09   1.15   1.09
SVR (linear)-DI      1.17   1.26   1.33   1.31   1.44   1.18   1.35  1.13   0.95   0.93
SVR (polynomial)-DI  1.17   1.23   1.33   1.24   1.17   1.31   1.36  1.32   1.05   1.02
SVR (rbf)-DI         1.06   1.08   1.11   1.12   1.18   1.03   1.03  1.02   1.04   1.04
SVR (sigmoid)-DI     1.22   1.20   1.23   1.27   1.30   1.22   1.15  0.98   0.99   0.95
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.30   1.39   1.37   1.42   1.36   1.22   1.28  1.13   1.18   1.09
XGBoost-DI           1.11   1.18   1.25   1.26   1.23   0.94   1.11  1.09   1.10   1.06
AdaBoost-DI          1.00   1.08   1.15   1.17   1.13   0.93   1.00  1.02   1.06   1.01
Gradient Boost-DI    1.28   1.34   1.37   1.34   1.32   1.10   1.24  1.12   1.08   1.09

62










Table 25: Canada/US Foreign Exchange Rate: relative RMSPE (sample period: 1960m1-2019m12)

    Pre-Pandemic Sample                                      NBER recession periods
Model                h=1    h=3    h=6    h=9    h=12   h=1    h=3    h=6    h=9    h=12
Baseline Model
RW (RMSPE)          0.173  0.117  0.088  0.073  0.064  0.228  0.161  0.118  0.095  0.082
Individual machine learning models
kNN (uniform)        1.03   1.04   1.09   1.10   1.10   1.02   1.06   1.16   1.20   1.10
kNN (inverse)        1.03   1.04   1.09   1.10   1.10   1.02   1.06   1.15   1.19   1.11
Decision Tree        1.38   1.42   1.43   1.37   1.35   1.28   1.51   1.13   1.00   0.97
SVR (linear)         1.26   1.37   1.53   1.51   1.42   1.38   1.62   1.57   1.61   1.50
SVR (polynomial)     1.12   1.32   1.32   1.14   1.17   1.43   1.57   1.02   1.06   1.03
SVR (rbf)            1.02   1.05   1.09   1.10   1.11   1.03   1.05   1.07   1.09   1.05
SVR (sigmoid)        1.17   1.16   1.22   1.28   1.26   1.26   1.17   1.05   1.12   1.12
Ensemble machine learning models
Random Forest        1.07   1.20   1.26   1.22   1.25   1.13   1.37   1.13   0.94   0.94
XGBoost              1.07   1.23   1.24   1.24   1.22   1.12   1.39   1.14   1.13   0.94
AdaBoost             1.01   1.14   1.23   1.21   1.29   0.98   1.31   1.10   1.02   0.98
Gradient Boost       1.28   1.41   1.37   1.34   1.37   1.22   1.55   1.12   0.99   0.97
Individual machine learning models using dimension reduction
kNN (uniform)-DI     1.03   1.11   1.15   1.11   1.09   1.07   1.14   1.07   1.09   1.00
kNN (inverse)-DI     1.03   1.11   1.16   1.11   1.09   1.07   1.14   1.07   1.08   1.00
Decision Tree-DI     1.39   1.39   1.43   1.40   1.39   1.19   1.33   1.31   1.49   1.20
SVR (linear)-DI      1.16   1.34   1.52   1.51   1.47   1.08   1.46   1.24   1.21   1.19
SVR (polynomial)-DI  1.27   1.49   1.90   1.58   1.47   1.62   2.03   1.70   1.45   1.44
SVR (rbf)-DI         1.07   1.14   1.15   1.17   1.19   1.03   1.08   1.13   1.13   1.06
SVR (sigmoid)-DI     1.16   1.36   1.37   1.42   1.42   1.17   1.44   1.25   1.08   0.97
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.37   1.41   1.46   1.41   1.38   1.12   1.33   1.32   1.48   1.21
XGBoost-DI           1.17   1.26   1.35   1.22   1.20   1.15   1.33   1.04   1.08   0.94
AdaBoost-DI          1.00   1.11   1.22   1.19   1.17   0.93   1.16   1.11   1.13   1.00
Gradient Boost-DI    1.31   1.40   1.45   1.35   1.35   1.21   1.41   1.28   1.39   1.27

63










Table 26: Industrial Production growth: RMSPE relative to ARDI (sample period: 1960m1-2019m12)

    Full out-of-sample
Model                h=1     h=3    h=6    h=9    h=12
Baseline Model
ARDI (RMSPE)        0.089   0.065  0.058  0.057  0.061
Individual machine learning models
kNN (uniform)       0.92**   0.93   0.89   0.83   0.72
kNN (inverse)       0.91**   0.93   0.89   0.83   0.72
Decision Tree        1.23    1.17   1.15   1.03   0.86
SVR (linear)         1.07    1.07   1.02   0.96   0.86
SVR (polynomial)     0.95    1.06   1.16   1.05   0.86
SVR (rbf)            0.93    0.93   0.90   0.86   0.77
SVR (sigmoid)        0.97    0.96   0.96   0.91   0.83
Ensemble machine learning models
Random Forest      0.90***   0.96   1.00   0.94   0.77
XGBoost             0.94*    0.93   0.96   0.93   0.78
AdaBoost           0.87***  0.88**  0.97   0.91   0.75
Gradient Boost       1.16    1.14   1.16   1.01   0.85
Individual machine learning models using dimension reduction
kNN (uniform)-DI     0.94    0.90   0.85   0.82   0.73
kNN (inverse)-DI     0.94    0.90   0.85   0.82   0.73
Decision Tree-DI     1.13    1.15   1.17   1.11   0.93
SVR (linear)-DI      1.04    1.30   1.29   1.12   1.04
SVR (polynomial)-DI  1.77    1.40   1.04   1.04   0.95
SVR (rbf)-DI         0.98    1.03   0.98   0.89   0.78
SVR (sigmoid)-DI     1.17    1.18   1.16   1.23   1.05
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.12    1.16   1.17   1.12   0.93
XGBoost-DI           1.05    0.94   0.96   0.94   0.84
AdaBoost-DI      0.89***    0.90**  0.96   0.92   0.81
Gradient Boost-DI    1.13    1.13   1.16   1.09   0.91

64










Table 27: Employment: RMSPE relative to ARDI (sample period: 1960m1-2019m12)

    Full out-of-sample
Model                h=1     h=3    h=6    h=9    h=12
Baseline Model
ARDI (RMSPE)        0.019   0.015  0.015  0.016  0.017
Individual machine learning models
kNN (uniform)        0.98    1.11   1.08   1.09   1.01
kNN (inverse)        0.98    1.11   1.08   1.09   1.01
Decision Tree        1.13    1.20   1.30   1.21   1.12
SVR (linear)         1.02    1.08   1.04   1.03   0.98
SVR (polynomial)     1.11    1.46   1.45   1.38   1.23
SVR (rbf)            0.98    1.10   1.04   1.05   0.98
SVR (sigmoid)        0.97    1.06   1.06   1.02   0.97
Ensemble machine learning models
Random Forest      0.83***   0.97   1.05   1.02   0.97
XGBoost            0.86***   0.91   0.96   0.97   0.94
AdaBoost           0.84***   0.91   1.03   1.06   0.98
Gradient Boost       1.10    1.12   1.26   1.19   1.10
Individual machine learning models using dimension reduction
kNN (uniform)-DI    0.90**   0.90   0.84   0.88   0.85
kNN (inverse)-DI   0.90***   0.89   0.85   0.88   0.85
Decision Tree-DI     1.11    1.12   1.19   1.16   1.11
SVR (linear)-DI    0.90***  0.85**  0.88   0.95   0.97
SVR (polynomial)-DI  2.78    4.02   4.06   3.74   3.18
SVR (rbf)-DI         0.98    1.11   1.05   1.06   1.00
SVR (sigmoid)-DI     3.30    3.48   2.81   2.57   2.23
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.12    1.13   1.18   1.17   1.11
XGBoost-DI         0.89***   0.89   0.90   0.95   0.91
AdaBoost-DI        0.85***  0.88*   0.91   0.95   0.92
Gradient Boost-DI    1.06    1.09   1.16   1.16   1.10

65










Table 28: Real Personal Income: RMSPE relative to ARDI (sample period: 1960m1-2019m12)

    Full out-of-sample
Model                h=1      h=3      h=6    h=9    h=12
Baseline Model
ARDI (RMSPE)        0.091    0.044    0.029  0.027  0.028
Individual machine learning models
kNN (uniform)      0.80***   0.89*     0.94   0.91   0.82
kNN (inverse)      0.80***   0.89*     0.94   0.91   0.82
Decision Tree        1.03     1.18     1.14   1.03   0.98
SVR (linear)         0.95     1.05     1.09   1.04   0.93
SVR (polynomial)   0.79***   0.88**    1.00   1.03   0.92
SVR (rbf)          0.76***  0.82***  0.86***  0.84   0.78
SVR (sigmoid)      0.77***  0.84***   0.89**  0.90   0.85
Ensemble machine learning models
Random Forest      0.82***   0.88**    0.92   0.88   0.83
XGBoost             0.89*     0.92    0.90*   0.86   0.79
AdaBoost           0.76***  0.82*** 0.82***  0.80*   0.73
Gradient Boost       0.98     1.05     1.10   1.01   0.97
Individual machine learning models using dimension reduction
kNN (uniform)-DI   0.78***   0.86**    0.89   0.87   0.82
kNN (inverse)-DI   0.78***   0.86**    0.89   0.87   0.82
Decision Tree-DI     0.93     1.04     1.08   0.97   0.89
SVR (linear)-DI    0.80***    0.94     1.06   1.05   0.98
SVR (polynomial)-DI  1.21     1.60     1.75   1.82   1.65
SVR (rbf)-DI       0.77***  0.85***    0.92   0.89   0.83
SVR (sigmoid)-DI     0.99     1.05     1.08   1.01   0.94
Ensemble machine learning models using dimension reduction
Random Forest-DI     0.94     1.04     1.08   0.96   0.88
XGBoost-DI          0.90*    0.90*     0.92   0.90   0.80
AdaBoost-DI        0.82***  0.83***   0.88**  0.82   0.75
Gradient Boost-DI    0.95     0.98     1.02   0.94   0.90

66










Table 29: Unemployment Rate: RMSPE relative to ARDI (sample period: 1960m1-2019m12)

    Full out-of-sample
Model                h=1     h=3    h=6    h=9    h=12
Baseline Model
ARDI (RMSPE)        2.154   1.311  1.143  1.118  1.161
Individual machine learning models
kNN (uniform)      0.91***   0.96   0.94   0.90   0.84
kNN (inverse)      0.91***   0.96   0.94   0.90   0.84
Decision Tree        1.20    1.22   1.28   1.14   1.07
SVR (linear)         1.03    1.07   1.01   0.97   0.94
SVR (polynomial)     0.95    1.16   1.25   1.15   1.03
SVR (rbf)           0.91**   0.96   0.97   0.96   0.90
SVR (sigmoid)       0.93*    0.95   0.98   0.94   0.87
Ensemble machine learning models
Random Forest      0.89***   0.97   1.08   1.01   0.92
XGBoost             0.94*    0.94   1.04   0.95   0.90
AdaBoost           0.86***   0.90   0.97   0.94   0.86
Gradient Boost       1.16    1.16   1.29   1.10   1.04
Individual machine learning models using dimension reduction
kNN (uniform)-DI     0.95    0.91   0.89   0.89   0.85
kNN (inverse)-DI     0.95    0.91   0.89   0.89   0.85
Decision Tree-DI     1.15    1.15   1.15   1.08   1.09
SVR (linear)-DI      1.10    1.08   1.04   1.06   1.02
SVR (polynomial)-DI  1.10    1.00   1.17   1.13   1.12
SVR (rbf)-DI         0.98    1.03   1.03   1.00   0.94
SVR (sigmoid)-DI     1.11    1.07   1.14   1.06   1.06
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.15    1.12   1.14   1.08   1.09
XGBoost-DI           1.00    0.96   0.95   0.99   0.99
AdaBoost-DI        0.90***  0.89**  0.88  0.90**  0.91
Gradient Boost-DI    1.16    1.14   1.10   1.06   1.06

67










Table 30: Real PCE: RMSPE relative to ARDI (sample period: 1960m1-2019m12)

    Full out-of-sample
Model                h=1     h=3     h=6    h=9    h=12
Baseline Model
ARDI (RMSPE)         0.07   0.034   0.025  0.026  0.025
Individual machine learning models
kNN (uniform)      0.85***  0.86*    0.88   0.78   0.80
kNN (inverse)      0.85***  0.86*    0.88   0.78   0.80
Decision Tree        1.14    1.09    1.00   0.86   0.88
SVR (linear)         1.01    0.97    0.98   0.85   0.88
SVR (polynomial)   0.86***   0.92    0.98   0.86   0.86
SVR (rbf)          0.84***  0.82**   0.82   0.76   0.77
SVR (sigmoid)      0.91***  0.86**   0.85   0.78   0.79
Ensemble machine learning models
Random Forest      0.87***  0.86**   0.83   0.74   0.76
XGBoost            0.88***   0.91    0.82   0.78   0.79
AdaBoost           0.84***  0.84**  0.80*   0.76   0.76
Gradient Boost      1.07     1.06    0.98   0.85   0.87
Individual machine learning models using dimension reduction
kNN (uniform)-DI   0.89***   0.88    0.85   0.78   0.79
kNN (inverse)-DI   0.89***   0.88    0.85   0.78   0.79
Decision Tree-DI     1.09    1.07    1.00   0.92   0.91
SVR (linear)-DI      0.96    0.98    1.00   0.94   0.94
SVR (polynomial)-DI  1.07    1.10    1.00   0.91   0.95
SVR (rbf)-DI      0.89***    0.91    0.86   0.77   0.78
SVR (sigmoid)-DI    0.94*    0.91    0.95   0.81   0.83
Ensemble machine learning models using dimension reduction
Random Forest-DI     1.11    1.08    1.00   0.92   0.92
XGBoost-DI           0.95    0.92    0.85   0.79   0.81
AdaBoost-DI        0.84***  0.84**   0.82   0.76   0.77
Gradient Boost-DI    1.03    1.07    1.00   0.89   0.90

10  Industrial Production  4  Employment

2

5  0

Percentage Change  0  Percentage Change  -2
-4

-6
-5
-8

-10  -10

-12

-15  2008 2010 2012 2014 2016 2018 2020 2022  -14  2008 2010 2012 2014 2016 2018 2020 2022
Date  Date



                      25    Real Personal Income                       12    Unemployment Rate

                      20                                               10

                      15                                                8

  Percentage Change   1050        Change (pp)                         642

                      -5                                                0

                      -10                                              -2

                      -15     2008 2010 2012 2014 2016 2018 2020 2022  -4    2008 2010 2012 2014 2016 2018 2020 2022
                                    Date        Date



  10                  Real Personal Consumption Expenditure




                      5


  Percentage Change   -50


  -10




  -15                     2008 2010 2012 2014     2016 2018 2020 2022
                                              Date

                                              Figure 8: Real variables extreme observations





                                              68