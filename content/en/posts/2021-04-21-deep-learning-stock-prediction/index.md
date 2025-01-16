---
title: "Research on Stock Price Prediction and Quantitative Strategies Based on Deep Learning"
date: 2021-04-21T12:00:00+08:00
author: "Yue Shui"
tags: ["Deep learning", "AI", "RNN", "LSTM", "BiLSTM", "GRU", "LightGBM", "Neural Networks", "Stock Prediction", "Financial Modeling", "Machine Learning", "Quantitative Investment", "Portfolio Management", "Financial Engineering", "Algorithmic Trading", "Time Series"]
categories: ["Technical Blog"]
toc: true
ShowToc: true
TocOpen: false
draft: false
math: true
---

## Abstract

The stock market is a crucial component of the financial market. In recent years, the stock market has flourished, attracting researchers from various fields to study stock price prediction and quantitative investment strategies. With the development of artificial intelligence and machine learning in recent years, scholars have transitioned from traditional statistical models to AI algorithms. Particularly, after the surge of deep learning, neural networks have achieved commendable results in stock price prediction and quantitative investment strategy research. The goal of deep learning is to learn multi-level features by combining low-level features to construct abstract high-level features, thereby mining the distributed feature representations of data and performing complex nonlinear modeling based on this to accomplish prediction tasks. Among these, RNNs are widely applied to sequential data such as natural language and speech. Daily stock prices and trading information are sequential data; therefore, many researchers have previously used RNNs to predict stock prices. However, basic recurrent neural networks suffer from the vanishing gradient problem when the number of layers is too large, a problem that the emergence of LSTM solved. Subsequently, variants of LSTM such as GRU, Peephole LSTM, and BiLSTM appeared. Traditional stock prediction models often overlook the temporal factors or only consider unidirectional temporal relationships. Therefore, this paper employs the BiLSTM model for stock price prediction. From a theoretical perspective, the BiLSTM model fully utilizes the contextual relationships in both forward and backward temporal directions in time series, avoiding the vanishing and exploding gradient problems in long sequences, and better learning information with long-term dependencies.

The first part of this paper's experiment establishes stock prediction models using LSTM, GRU, and BiLSTM based on stock data from China’s Shanghai Pudong Development Bank and the foreign company IBM, respectively. By comparing the prediction results of these three deep learning models, it was found that the BiLSTM model outperforms the other models on both datasets, achieving higher prediction accuracy. The second part uses stock data from the entire A-share market and first applies the LightGBM model to screen 50 factors, selecting the top 10 most important factors. Subsequently, the BiLSTM model is used to select and combine these factors to establish a quantitative investment strategy. Finally, empirical tests and backtesting demonstrate that this strategy outperforms the market benchmark index, illustrating the practical application value of the BiLSTM model in stock price prediction and quantitative investment.

**Keywords**: Quantitative Investment; Deep Learning; Neural Network Models; Multi-Factor Stock Selection; BiLSTM

## Chapter 1: Introduction

### 1.1 Research Background and Significance

#### 1.1.1 Research Background

Since its gradual emergence in the 1970s, quantitative investment has entered the view of various investors, initiating a new revolution that changed the previously dominated landscape of passive management and the Efficient Market Hypothesis (EMH) in portfolio management. The EMH posits that under the premise of market efficiency, stock prices reflect all available information, and no excess returns exist. Passive investment management is based on the belief that markets are efficient, focusing more on asset classes. The most common method is purchasing index funds and tracking the performance of published indices. In contrast, active investment management primarily relies on investors' subjective judgments about the market and individual stocks, using publicly available data and applying mathematical models in the financial field to evaluate stocks and construct investment portfolios to achieve returns. Quantitative investment, which involves statistical processing of large amounts of historical data to uncover investment opportunities and eliminate subjective factors, has been increasingly favored by investors. Since the rise of quantitative investment, various technologies have been employed to predict stock prices better and establish quantitative investment strategies. Early scholars domestically and internationally used statistical methods for modeling stock price prediction, such as the moving average method, multiple regression, and autoregressive integrated moving average (ARIMA) models. However, due to the numerous factors influencing the stock market and the large volume of data, stock prediction is highly challenging, and the predictive performance of these statistical models is often unsatisfactory.

In recent years, related technologies like machine learning, deep learning, and neural networks have continuously developed, providing technical support for stock price prediction and the construction of quantitative strategies. Many scholars have achieved breakthroughs using methods such as random forests, neural networks, support vector machines, and convolutional neural networks. The ample historical data in the stock market, combined with diverse technical support, provides favorable conditions for stock price prediction and the construction of quantitative strategies.

#### 1.1.2 Research Significance

From the perspective of the national economic system and the long-term development of financial markets, research on stock price prediction models and quantitative investment strategies is indispensable. China started relatively late, and its financial markets are not yet mature, with insufficient financial instruments and lower market efficiency. However, in recent years, the government has gradually relaxed policies and vigorously developed financial markets, providing a "fertile ground" for the development of quantitative investment and emerging financial technologies. Developing quantitative investment and new financial technologies can offer opportunities for China's financial markets to achieve leapfrog growth. Moreover, stock price indices, as important economic indicators, act as barometers for China's economic development.

From the perspective of individual and institutional investors, constructing stock price prediction models and quantitative investment strategy models enhances market efficiency. Individual investors often lack sufficient professional knowledge, leading to somewhat blind investment behaviors. Constructing relevant models to provide references can reduce the probability of judgment errors and change the relatively disadvantaged position of individual investors in the capital market. For institutional investors, rational and objective models combined with personal judgment improve decision accuracy and provide new directions for investment activities.

In summary, considering China's current development status, this paper selects individual stocks for stock price prediction models and the entire A-share market for quantitative strategy research, which holds significant practical research significance.

### 1.2 Literature Review

[White (1988)](https://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf)$^{[1]}$ used BP neural networks to predict IBM’s daily returns. However, due to the BP neural network model being susceptible to gradient explosion, the model could not converge to the global minimum, resulting in inaccurate predictions.

[Kimoto (1990)](https://web.ist.utl.pt/adriano.simoes/tese/referencias/Papers%20-%20Adriano/NN.pdf)$^{[2]}$ developed a system for forecasting the Tokyo Stock Exchange Prices Indexes (TOPIX) using modular neural network technology. The system not only successfully predicted TOPIX but also achieved a certain level of profitability through simulated stock trading based on the prediction results.

[G. Peter Zhang (2003)](https://dl.icdst.org/pdfs/files/2c442c738bd6bc178e715f400bec5d5f.pdf)$^{[3]}$ conducted a comparative study on the performance of the Autoregressive Integrated Moving Average (ARIMA) model and the Artificial Neural Network (ANN) model in time series forecasting. The results showed that the ANN model significantly outperformed the ARIMA model in time series prediction accuracy.

[Ryo Akita (2016)](https://ieeexplore.ieee.org/document/7550882)$^{[4]}$ selected consumer price indices, price-earnings ratios, and various events from newspapers as features. Using paragraph vectors and LSTM networks, a financial time series prediction model was constructed. Empirical data from fifty listed companies on the Tokyo Stock Exchange validated the model's effectiveness in predicting stock opening prices.

[Kunihiro Miyazaki (2017)](https://www.ai-gakkai.or.jp/jsai2017/webprogram/2017/pdf/1112.pdf)$^{[5]}$ constructed a model for predicting the rise and fall of the Topix Core 30 index and its constituent stocks by extracting daily stock chart images and 30-minute stock price data. The study compared various models, including Logistic Regression (LR), Random Forest (RF), Multilayer Perceptron (MLP), LSTM, CNN, PCA-CNN, and CNN-LSTM. The results indicated that LSTM had the best predictive performance, CNN performed moderately, and hybrid models combining CNN and LSTM could enhance prediction accuracy.

[Taewook Kim (2019)](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0212320&type=printable)$^{[6]}$ proposed a hybrid LSTM-CNN model that combines features from stock price time series and stock price images to predict the S&P 500 index. The study demonstrated that the LSTM-CNN model outperformed single models in stock price prediction and that such predictions hold practical significance for constructing quantitative investment strategies.

### 1.3 Innovations of the Paper

This paper presents the following innovations in stock prediction:

1. **Diverse Market Data Usage**: It separately uses data from China’s A-share Shanghai Pudong Development Bank and the foreign stock IBM, avoiding the limitations of studying a single market. Traditional BP models have never considered temporal factors, and models like LSTM only consider unidirectional temporal relationships. Therefore, this paper employs the BiLSTM model for stock price prediction. Theoretically, the BiLSTM model fully utilizes contextual relationships in both forward and backward temporal directions in time series, avoiding the vanishing and exploding gradient problems in long sequences, and better learning information with long-term dependencies. Empirical comparisons with LSTM and GRU models demonstrate its ability to enhance prediction accuracy.

2. **Multi-Feature Stock Price Prediction**: The stock price prediction model trains on multiple features, including opening price, closing price, highest price, and trading volume. Compared to single-feature prediction of closing prices, this approach is theoretically more accurate and better facilitates the comparison of prediction effects among LSTM, GRU, and BiLSTM models.

In terms of quantitative strategy research, this paper presents the following innovations:

1. **Custom Factor Selection**: Instead of using commonly available market factors, it employs multiple price and volume factors derived through Genetic Programming (GP) and manual data mining. The LightGBM model is then used to select the top 10 most important factors from an initial set of 50 factors.

2. **BiLSTM-Based Factor Combination**: Traditional quantitative investment models generally use LSTM and CNN models to build investment strategies. This paper utilizes data from the entire A-share market and employs the BiLSTM model for factor combination to establish a quantitative investment strategy. Empirical tests and backtesting show that this strategy outperforms the market benchmark index (CSI All Share Index), indicating the practical application value of the BiLSTM model in stock price prediction and quantitative investment.

### 1.4 Research Framework

This paper conducts stock price prediction and quantitative strategy research based on deep learning algorithms. The specific research framework is shown in Fig. 1:

{{< figure 
    src="Research Framework.svg" 
    caption="Fig. 1. Research Framework." 
    align="center" 
>}}

The specific research framework content is as follows:

- **Chapter 1: Introduction**. This chapter first introduces the research significance and background of stock price prediction and quantitative strategy research. It then reviews the current research status, outlines the innovations of this paper compared to existing research, and briefly describes the research framework.

- **Chapter 2: Theoretical Foundations**. This chapter introduces the deep learning models and the fundamental theories of quantitative stock selection involved in this research. The subsection on deep learning models sequentially introduces RNN, LSTM, GRU, and BiLSTM, with a focus on the internal structure of the LSTM model. The subsection on quantitative stock selection theory sequentially introduces the Mean-Variance Model, Capital Asset Pricing Model (CAPM), Arbitrage Pricing Theory (APT), Multi-Factor Models, and the Fama-French Three-Factor and Five-Factor Models. This subsection outlines the development of multi-factor quantitative stock selection from various financial theories and model development trajectories.

- **Chapter 3: Comparative Study of LSTM, GRU, and BiLSTM in Stock Price Prediction**. This chapter first introduces the domestic and foreign stock datasets used in the experiments, followed by data normalization and preprocessing steps. It then describes the network structures, model compilation, and hyperparameter settings of the LSTM, GRU, and BiLSTM models used in this chapter, and presents experimental results. Finally, it analyzes the experimental results and concludes the chapter.

- **Chapter 4: Research on LightGBM-BiLSTM-Based Quantitative Investment Models**. This chapter first provides an overview of the experimental steps, followed by a detailed introduction of the stock and factor data used in the experiments. It then performs factor cleaning steps, including handling missing values, outlier removal, factor normalization, and factor neutralization. Subsequently, it uses the LightGBM model for factor selection and the BiLSTM model for factor combination. Finally, it constructs a quantitative investment strategy based on the obtained models and conducts backtesting.

- **Chapter 5: Conclusion and Outlook**. This chapter summarizes the main research content regarding stock price prediction and quantitative investment strategies. It then discusses the current research limitations and provides prospects for future research directions.

## Chapter 2: Theoretical Foundations

### 2.1 Deep Learning Models

#### 2.1.1 RNN

Recurrent Neural Networks (RNNs) are widely applied to sequential data such as natural language and speech. Daily stock prices and trading information are sequential data; therefore, many previous studies have used RNNs to predict stock prices. RNNs employ a simple repetitive module in a chain-like structure, such as a single tanh layer. However, basic RNNs encounter the vanishing gradient problem when the number of layers becomes too large, a problem that the emergence of LSTM resolved. Fig. 2 is the RNN structure diagram.

{{< figure 
    src="RNN.png" 
    caption="Fig. 2. RNN Structure Diagram. (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))" 
    align="center" 
>}}

#### 2.1.2 LSTM

Long Short-Term Memory (LSTM) networks are a special type of RNN capable of learning long-term dependencies. They were proposed by [Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)$^{[7]}$ and have been improved and popularized by many subsequent works. Due to their unique design structure, LSTMs are relatively insensitive to sequence length and solve the vanishing and exploding gradient problems inherent in traditional RNNs. Compared to traditional RNNs and other time series models like Hidden Markov Models (HMMs), LSTMs can better handle and predict significant events in time series with long intervals and delays. Consequently, LSTMs are widely used in machine translation and speech recognition fields.

LSTMs are explicitly designed to avoid the long-term dependency problem. All recurrent neural networks have a chain-like structure of repeated network modules, whereas LSTMs modify the RNN structure. Instead of using a single neural network layer, LSTMs use a special four-layer structure that interacts in a unique way.

{{< figure 
    src="LSTM.png" 
    caption="Fig. 3. LSTM Structure Diagram 1. (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))" 
    align="center" 
>}}

{{< figure 
    src="LSTM2.png" 
    caption="Fig. 4. LSTM Structure Diagram 2. (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))" 
    align="center" 
>}}

As shown in Fig. 3, the black lines represent the transmission of a node's output vector to another node's input vector. The Neural Network Layer is a processing module with a $\sigma$ activation function or tanh activation function; Pointwise Operation represents element-wise multiplication between vectors; Vector Transfer indicates the direction of information transmission; Concatenate and Copy are represented by two black lines merging together and separating, respectively, indicating the merging and copying of information.

Below, we detail the structure of an LSTM using the following components:

1. **Forget Gate**

{{< figure 
    src="forget_gate.png" 
    caption="Fig. 5. Forget Gate Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))" 
    align="center" 
>}}

$$
f_{t} = \sigma\left(W_{f} \cdot \left[h_{t-1}, x_{t}\right] + b_{f}\right)
$$

**Parameter Description:**

- $h_{t-1}$: Output from the previous time step
- $x_{t}$: Input at the current time step
- $\sigma$: Sigmoid activation function
- $W_{f}$: Weight matrix for the forget gate
- $b_{f}$: Bias vector parameter for the forget gate

The first step, as shown in Fig. 5, is the process of deciding which information to discard from the cell state. This is determined by computing $f_{t}$ using the sigmoid function (where $f_{t}$ ranges between 0 and 1, with 0 meaning completely blocked and 1 meaning completely passed) to decide whether to allow the previous cell state $C_{t-1}$ to pass through or be partially allowed.

2. **Input Gate**

{{< figure 
    src="input_gate1.png" 
    caption="Fig. 6. Input Gate Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)) " 
    align="center" 
>}}

$$
\begin{aligned}
i_{t} &= \sigma\left(W_{i} \cdot \left[h_{t-1}, x_{t}\right] + b_{i}\right) \\
\tilde{C}_{t} &= \tanh\left(W_{C} \cdot \left[h_{t-1}, x_{t}\right] + b_{C}\right)
\end{aligned}
$$

**Parameter Description:**

- $h_{t-1}$: Output from the previous time step
- $x_{t}$: Input at the current time step
- $\sigma$: Sigmoid activation function
- $W_{i}$: Weight matrix for the input gate
- $b_{i}$: Bias vector parameter for the input gate
- $W_{C}$: Weight matrix for the cell state
- $b_{C}$: Bias vector parameter for the cell state
- $\tanh$: Tanh activation function

The second step, as shown in Fig. 6, involves using the sigmoid function to determine what information to store in the cell state. Subsequently, a $\tanh$ layer creates the candidate vector $\tilde{C}_{t}$, which will be added to the cell state.

{{< figure 
    src="input_gate2.png" 
    caption="Fig. 7. Current Cell State Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))" 
    align="center" 
>}}

$$
C_{t} = f_{t} * C_{t-1} + i_{t} * \tilde{C}_{t}
$$

**Parameter Description:**

- $C_{t-1}$: Cell state from the previous time step
- $\tilde{C}_{t}$: Candidate cell state
- $i_{t}$: Value of the input gate
- $f_{t}$: Value of the forget gate

The third step, as shown in Fig. 7, calculates the current cell state $C_t$ by combining the effects of the forget gate and the input gate.
- The forget gate $f_t$ weights the previous cell state $C_{t-1}$ to discard unnecessary information.
- The input gate $i_t$ weights the candidate cell state $\tilde{C}_t$ to determine how much new information to introduce.
Finally, the two parts are added together to update the current cell state $C_t$.

3. **Output Gate**

{{< figure 
    src="output_gate.png" 
    caption="Fig. 8. Output Gate Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))" 
    align="center" 
>}}

$$
\begin{aligned}
o_{t} &= \sigma\left(W_{o} \cdot \left[h_{t-1}, x_{t}\right] + b_{o}\right) \\
h_{t} &= o_{t} * \tanh\left(C_{t}\right)
\end{aligned}
$$

**Parameter Description:**

- $o_{t}$: Value of the output gate
- $\sigma$: Sigmoid activation function
- $W_{o}$: Weight matrix for the output gate
- $h_{t-1}$: Output from the previous time step
- $x_{t}$: Input at the current time step
- $b_{o}$: Bias vector parameter for the output gate
- $h_{t}$: Output at the current time step
- $\tanh$: Tanh activation function
- $C_{t}$: Current cell state

The fourth step, as shown in Fig. 8, uses the sigmoid function to calculate the output gate's value. Finally, the cell state $C_t$ at this time step is processed through the tanh activation function and multiplied by the output gate's value $o_t$ to obtain the current output $h_t$.

#### 2.1.3 GRU

[K. Cho (2014)](https://arxiv.org/abs/1406.1078)$^{[8]}$ proposed the Gated Recurrent Unit (GRU). GRU primarily simplifies and adjusts LSTM by merging LSTM's original forget gate, input gate, and output gate into an update gate and a reset gate. Additionally, GRU combines the cell state with the hidden state, reducing model complexity while still achieving performance comparable to LSTM on certain tasks.

This model can save considerable time when training on larger datasets and demonstrates better performance on some smaller and less frequent datasets$^{[9][10]}$.

{{< figure 
    src="GRU.png" 
    caption="Fig. 9. GRU Structure Diagram (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))" 
    align="center" 
>}}

$$
\begin{aligned}
z_{t} &= \sigma\left(W_{z} \cdot \left[h_{t-1}, x_{t}\right]\right) \\
r_{t} &= \sigma\left(W_{r} \cdot \left[h_{t-1}, x_{t}\right]\right) \\
\tilde{h}_{t} &= \tanh\left(W \cdot \left[r_{t} * h_{t-1}, x_{t}\right]\right) \\
h_{t} &= \left(1 - z_{t}\right) * h_{t-1} + z_{t} * \tilde{h}_{t}
\end{aligned}
$$

**Parameter Description:**

- $z_{t}$: Value of the update gate
- $r_{t}$: Value of the reset gate
- $W_{z}$: Weight matrix for the update gate
- $W_{r}$: Weight matrix for the reset gate
- $\tilde{h}_{t}$: Candidate hidden state

#### 2.1.4 BiLSTM

Bidirectional Long Short-Term Memory (BiLSTM) networks combine forward and backward LSTMs. The BiLSTM model fully utilizes the contextual relationships in both forward and backward temporal directions in time series, enabling the learning of information with long-term dependencies. Compared to unidirectional LSTM, BiLSTM can better consider the reverse influence of future data. Fig. 10 is the BiLSTM structure diagram.

{{< figure 
    src="BiLSTM.png" 
    caption="Fig. 10. BiLSTM Structure Diagram. (Image source: [Baeldung](https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm))" 
    align="center" 
>}}

### 2.2 Quantitative Stock Selection Theory

#### 2.2.1 Mean-Variance Model

The quantitative stock selection strategy originated in the 1950s when [Markowitz (1952)](https://www.jstor.org/stable/2975974)$^{[11]}$ proposed the Mean-Variance Model. This model not only laid the foundation for modern portfolio theory by quantifying investment risk but also established a specific model describing risk and expected return. It broke the previous situation of only qualitative analysis of portfolios without quantitative analysis, successfully introducing mathematical models into the financial investment field.

$$
\begin{aligned}
\mathrm{E}\left(R_{p}\right) &= \sum_{i=1}^{n} w_{i} \mathrm{E}\left(R_{i}\right) \\
\sigma_{p}^{2} &= \sum_{i=1}^{n} \sum_{j=1}^{n} w_{i} w_{j} \operatorname{Cov}\left(R_{i}, R_{j}\right) = \sum_{i=1}^{n} \sum_{j=1}^{n} w_{i} w_{j} \sigma_{i} \sigma_{j} \rho_{ij} \\
\sigma_{i} &= \sqrt{\operatorname{Var}\left(R_{i}\right)}, \quad \rho_{ij} = \operatorname{Corr}\left(R_{i}, R_{j}\right)
\end{aligned}
$$

$$
\min \sigma_{p}^{2} \quad \text{subject to} \quad \sum_{i=1}^{n} \mathrm{E}\left(R_{i}\right) w_{i} = \mu_{p}, \quad \sum_{i=1}^{n} w_{i} = 1
$$

$$
\begin{aligned}
\Omega &= \begin{pmatrix}
\sigma_{11} & \cdots & \sigma_{1n} \\
\vdots & \ddots & \vdots \\
\sigma_{n1} & \cdots & \sigma_{nn}
\end{pmatrix} = \begin{pmatrix}
\operatorname{Var}\left(R_{1}\right) & \cdots & \operatorname{Cov}\left(R_{1}, R_{n}\right) \\
\vdots & \ddots & \vdots \\
\operatorname{Cov}\left(R_{n}, R_{1}\right) & \cdots & \operatorname{Var}\left(R_{n}\right)
\end{pmatrix} \\
\Omega^{-1} &= \begin{pmatrix}
v_{11} & \cdots & v_{1n} \\
\vdots & \ddots & \vdots \\
v_{n1} & \cdots & v_{nn}
\end{pmatrix} \\
w_{i} &= \frac{1}{D}\left(\mu_{p} \sum_{j=1}^{n} v_{ij}\left(C \mathrm{E}\left(R_{j}\right) - A\right) + \sum_{j=1}^{n} v_{ij}\left(B - A \mathrm{E}\left(R_{j}\right)\right)\right), \quad i = 1, \ldots, n
\end{aligned}
$$

$$
\begin{aligned}
A &= \sum_{i=1}^{n} \sum_{j=1}^{n} v_{ij} \mathrm{E}\left(R_{j}\right), \quad B = \sum_{i=1}^{n} \sum_{j=1}^{n} v_{ij} \mathrm{E}\left(R_{i}\right) \mathrm{E}\left(R_{j}\right), \quad C = \sum_{i=1}^{n} \sum_{j=1}^{n} v_{ij}, \quad D = BC - A^{2} > 0 \\
\sigma_{p}^{2} &= \frac{C \mu_{p}^{2} - 2A \mu_{p} + B}{D}
\end{aligned}
$$

**Where:**

- $\mathrm{E}\left(R_{p}\right)$ and $\mu_{p}$ are the expected return of portfolio $p$
- $\mathrm{E}\left(R_{i}\right)$ is the expected return of asset $i$
- $\sigma_{i}$ is the standard deviation of asset $i$
- $\sigma_{j}$ is the standard deviation of asset $j$
- $w_{i}$ is the weight of asset $i$ in the portfolio
- $\sigma_{p}^{2}$ is the variance of portfolio $p$
- $\rho_{ij}$ is the correlation coefficient between assets $i$ and $j$

Using the above formulas$^{[12]}$, we can construct an investment portfolio that minimizes non-systematic risk under a given expected return condition.

#### 2.2.2 Capital Asset Pricing Model

[William Sharpe (1964)](https://www.jstor.org/stable/2977928)$^{[13]}$, [John Lintner (1965)](https://www.jstor.org/stable/1924119)$^{[14]}$, and [Jan Mossin (1966)](https://www.jstor.org/stable/1910098)$^{[15]}$ proposed the Capital Asset Pricing Model (CAPM). This model posits that the expected return of an asset is related to its risk measure, the $\beta$ value. The model connects the expected return rate of an asset with market risk through a simple linear relationship, making Markowitz's (1952) portfolio selection theory more applicable to the real world and laying the theoretical foundation for multi-factor stock selection models.

According to the CAPM, for a given asset $i$, the relationship between its expected return and the expected return of the market portfolio can be expressed as:

$$
E\left(r_{i}\right) = r_{f} + \beta_{im}\left[E\left(r_{m}\right) - r_{f}\right]
$$

**Where:**

- $E\left(r_{i}\right)$ is the expected return of asset $i$
- $r_{f}$ is the risk-free rate
- $\beta_{im}$ (Beta) is the systematic risk coefficient of asset $i$, $\beta_{im} = \frac{\operatorname{Cov}\left(r_{i}, r_{m}\right)}{\operatorname{Var}\left(r_{m}\right)}$
- $E\left(r_{m}\right)$ is the expected return of the market portfolio $m$
- $E\left(r_{m}\right) - r_{f}$ is the market risk premium factor

#### 2.2.3 Arbitrage Pricing Theory and Multi-Factor Models

[Ross (1976)](https://www.top1000funds.com/wp-content/uploads/2014/05/The-Arbitrage-Theory-of-Capital-Asset-Pricing.pdf)$^{[16]}$ proposed the Arbitrage Pricing Theory (APT). This theory asserts that arbitrage activities are the decisive factor in forming market equilibrium prices. By introducing a series of factors into the process of return formation and constructing linear relationships, APT overcomes the limitations of the Capital Asset Pricing Model (CAPM) and provides important theoretical guidance for subsequent researchers.

APT is considered the theoretical foundation of Multi-Factor Models (MFM), an essential component of asset pricing models and one of the cornerstones of asset pricing theory. The general expression of a multi-factor model is:

$$
r_{j} = a_{j} + \lambda_{j1} f_{1} + \lambda_{j2} f_{2} + \cdots + \lambda_{jn} f_{n} + \delta_{j}
$$

**Where:**

- $r_{j}$ is the return of asset $j$
- $a_{j}$ is the constant term for asset $j$
- $f_{n}$ are the systematic factors
- $\lambda_{jn}$ are the factor loadings
- $\delta_{j}$ is the random error

#### 2.2.4 Fama-French Three-Factor and Five-Factor Models

[Fama (1992) and French (1992)](https://www.bauer.uh.edu/rsusmel/phd/Fama-French_JFE93.pdf)$^{[17]}$ discovered through cross-sectional regressions combined with time series analysis that a stock’s $\beta$ value cannot explain the differences in returns among different stocks. However, factors such as a company’s market capitalization, book-to-market ratio, and price-earnings ratio can significantly explain the differences in stock returns. They concluded that excess returns compensate for risk factors not captured by the $\beta$ in CAPM, leading to the Fama-French Three-Factor Model. These three factors are:

- **Market Risk Premium Factor**  
  - Represents the overall systematic risk of the market, calculated as the expected return of the market portfolio minus the risk-free rate.  
  - Measures the excess return investors expect for taking on systematic risk (risk that cannot be eliminated through diversification).  
  - Calculation formula:  
    $$
    \text{Market Risk Premium} = E(R_m) - R_f
    $$
    where $E(R_m)$ is the expected return of the market, and $R_f$ is the risk-free rate.
  
- **Size Factor (SMB: Small Minus Big)**  
  - Represents the return difference between small-cap and large-cap stocks.  
  - Small-cap stocks generally carry higher risk but historically show higher expected returns than large-cap stocks.  
  - Calculation formula:  
    $$
    SMB = R_{\text{Small}} - R_{\text{Big}}
    $$
    Reflects the market’s compensation for the additional risk premium of small-cap stocks.
  
- **Book-to-Market Ratio Factor (HML: High Minus Low)**  
  - Reflects the return difference between high book-to-market (value) stocks and low book-to-market (growth) stocks.  
  - High book-to-market stocks are typically undervalued by the market but may offer higher returns in the long run.  
  - Calculation formula:  
    $$
    HML = R_{\text{High}} - R_{\text{Low}}
    $$
    Low book-to-market stocks may be overvalued due to the market’s overly optimistic expectations.

This model specifies the factors in the APT model and concludes that investing in small-cap and high-growth stocks exhibits high-risk and high-return characteristics. The Fama-French Three-Factor Model is widely applied in modern investment behavior analysis and practice.

Subsequently, [Fama (2015) and French (2015)](https://tevgeniou.github.io/EquityRiskFactors/bibliography/FiveFactor.pdf)$^{[18]}$ expanded the three-factor model by adding two more factors:

- **Profitability Factor (RMW: Robust Minus Weak)**  
  - Reflects the return difference between high-profitability and low-profitability companies.  
  - Companies with strong profitability (high ROE, net profit margin) are more likely to provide stable and higher returns.  
  - Calculation formula:  
    $$
    RMW = R_{\text{Robust}} - R_{\text{Weak}}
    $$
  
- **Investment Factor (CMA: Conservative Minus Aggressive)**  
  - Reflects the return difference between conservative and aggressive investment companies.  
  - Aggressive companies (rapid expansion, high capital expenditure) usually carry greater operational risks, while conservative companies (steady capital expenditure) exhibit higher stability and returns.  
  - Calculation formula:  
    $$
    CMA = R_{\text{Conservative}} - R_{\text{Aggressive}}
    $$

The Fama-French Three-Factor Model formula is:

$$
R_i - R_f = \alpha_i + \beta_{i,m} \cdot (R_m - R_f) + \beta_{i,SMB} \cdot SMB + \beta_{i,HML} \cdot HML + \epsilon_i
$$

The Fama-French Five-Factor Model formula is:

$$
R_i - R_f = \alpha_i + \beta_{i,m} \cdot (R_m - R_f) + \beta_{i,SMB} \cdot SMB + \beta_{i,HML} \cdot HML + \beta_{i,RMW} \cdot RMW + \beta_{i,CMA} \cdot CMA + \epsilon_i
$$

**Where:**

- $R_i$: Expected return of stock $i$
- $R_f$: Risk-free rate
- $R_m$: Expected return of the market portfolio
- $R_m - R_f$: Market risk premium factor
- $SMB$: Size factor (Small Minus Big)
- $HML$: Book-to-market ratio factor (High Minus Low)
- $RMW$: Profitability factor (Robust Minus Weak)
- $CMA$: Investment factor (Conservative Minus Aggressive)
- $\beta_{i,*}$: Sensitivity of stock $i$ to the corresponding factor
- $\epsilon_i$: Regression residual

#### 2.2.5 Model Comparison Tables

##### Table 2.1: Comparison of Models

The following table summarizes the core content and factor sources of the **Mean-Variance Model**, **Capital Asset Pricing Model (CAPM)**, **Arbitrage Pricing Theory (APT)**, and **Fama-French Models**:

| **Model**                   | **Core Content**                                                                                   | **Factor Sources**                                     |
|-----------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| **Mean-Variance Model**     | Foundation of portfolio theory, optimizing portfolios through expected returns and covariance matrices. | Expected returns and covariance matrices of portfolio assets |
| **Capital Asset Pricing Model (CAPM)** | Explains asset returns through market risk factor ($\beta$), laying the theoretical foundation for multi-factor models. | Market factor $\beta$                                  |
| **Arbitrage Pricing Theory (APT)**      | Multi-factor framework allowing multiple economic variables (e.g., inflation rate, interest rate) to explain asset returns. | Multiple factors (macroeconomic variables like inflation rate, interest rate) |
| **Fama-French Three-Factor Model**      | Enhances asset return explanation by adding size and book-to-market ratio factors.               | Market factor, SMB (Size factor), HML (Book-to-Market ratio factor) |
| **Fama-French Five-Factor Model**       | Further improves asset pricing by adding profitability and investment factors on top of the three-factor model. | Market factor, SMB, HML, RMW (Profitability factor), CMA (Investment factor) |

The following table summarizes the advantages and disadvantages of these models:

##### Table 2.2: Comparison of Model Advantages and Disadvantages

| **Model**                   | **Advantages**                                                                                      | **Disadvantages**                                                                                 |
|-----------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Mean-Variance Model**     | Provides a systematic method for portfolio optimization, laying the foundation for modern investment theory. | Optimizes only for returns and variance, without explicitly identifying sources of risk compensation. |
| **Capital Asset Pricing Model (CAPM)** | Simple to use, connects return differences to market risk, providing a theoretical foundation for multi-factor models. | Assumes a single factor (market risk) determines returns, ignoring other systematic risk factors.  |
| **Arbitrage Pricing Theory (APT)**      | Allows multiple factors to explain asset returns, reducing reliance on single-factor assumptions and offering greater flexibility. | Does not specify specific factors, making it less practical and only providing a framework.         |
| **Fama-French Three-Factor Model**      | Significantly improves the explanatory power of asset returns by adding size and book-to-market ratio factors. | Ignores other factors such as profitability and investment behaviors.                              |
| **Fama-French Five-Factor Model**       | Further enhances asset pricing by incorporating profitability and investment factors, capturing key variables influencing asset returns more comprehensively. | Higher model complexity, greater data requirements, and potential omission of certain latent factors. |

## Chapter 3: Comparative Study of LSTM, GRU, and BiLSTM in Stock Price Prediction

### 3.1 Introduction of Experimental Data

Many domestic and foreign studies focus primarily on their own stock indices, with relatively few studies examining individual stocks from different markets. Moreover, few studies have compared the performance of LSTM, GRU, and BiLSTM models. Therefore, this paper selects China's A-share Shanghai Pudong Development Bank (Pudong Bank, code 600000) and the US stock International Business Machines Corporation (IBM) for research, enabling a more accurate comparison of the three models used. For Pudong Bank, stock data from January 1, 2008, to December 31, 2020, comprising 3,114 valid data points, were sourced from the Tushare Financial Data Platform. The features selected for this dataset include date, opening price, closing price, highest price, lowest price, and trading volume. Except for the date, which serves as the time series index, the other five features are used as independent variables. For IBM, stock data from January 2, 1990, to November 15, 2018, comprising 7,278 valid data points, were sourced from Yahoo Finance. The features selected for this dataset include date, opening price, highest price, lowest price, closing price, adjusted closing price, and trading volume. Except for the date, which serves as the time series index, the other six features are used as independent variables. In this experiment, the closing price is chosen as the target variable. Tables 3.1 and 3.2 display partial data from the two datasets.

#### Table 3.1: Partial Display of Pudong Bank Dataset

| Date       | Open  | Close | High  | Low   | Volume      | Code   |
|------------|-------|-------|-------|-------|-------------|--------|
| 2008-01-02 | 9.007 | 9.101 | 9.356 | 8.805 | 131,583.90  | 600000 |
| 2008-01-03 | 9.007 | 8.645 | 9.101 | 8.426 | 211,346.56  | 600000 |
| 2008-01-04 | 8659  | 9.009 | 9.111 | 8.501 | 139,249.67  | 600000 |
| 2008-01-07 | 8.970 | 9.515 | 9.593 | 8.953 | 228,043.01  | 600000 |
| 2008-01-08 | 9.551 | 9.583 | 9.719 | 9.517 | 161,255.31  | 600000 |
| 2008-01-09 | 9.583 | 9.663 | 9.772 | 9.432 | 102,510.92  | 600000 |
| 2008-01-10 | 9.701 | 9.680 | 9.836 | 9.602 | 217,966.25  | 600000 |
| 2008-01-11 | 9.670 | 10.467| 10.532| 9.670 | 231,544.21  | 600000 |
| 2008-01-14 | 10.367| 10.059| 10.433| 10.027| 142,918.39  | 600000 |
| 2008-01-15 | 10.142| 10.051| 10.389| 10.006| 161,221.52  | 600000 |

**Data Source**: [Tushare](https://github.com/waditu/tushare)

#### Table 3.2: Partial Display of IBM Dataset

| Date       | Open    | High    | Low     | Close   | Adj Close | Volume  |
|------------|---------|---------|---------|---------|-----------|---------|
| 1990-01-02 | 23.6875 | 24.5313 | 23.6250 | 24.5000 | 6.590755  | 7,041,600 |
| 1990-01-03 | 24.6875 | 24.8750 | 24.5938 | 24.7188 | 6.649599  | 9,464,000 |
| 1990-01-04 | 24.7500 | 25.0938 | 24.7188 | 25.0000 | 6.725261  | 9,674,800 |
| 1990-01-05 | 24.9688 | 25.4063 | 24.8750 | 24.9375 | 6.708448  | 7,570,000 |
| 1990-01-08 | 24.8125 | 25.2188 | 24.8125 | 25.0938 | 6.750481  | 4,625,200 |
| 1990-01-09 | 25.1250 | 25.3125 | 24.8438 | 24.8438 | 6.683229  | 7,048,000 |
| 1990-01-10 | 24.8750 | 25.0000 | 24.6563 | 24.7500 | 6.658009  | 5,945,600 |
| 1990-01-11 | 24.8750 | 25.0938 | 24.8438 | 24.9688 | 6.716855  | 5,905,600 |
| 1990-01-12 | 24.6563 | 24.8125 | 24.4063 | 24.4688 | 6.582347  | 5,390,800 |
| 1990-01-15 | 24.4063 | 24.5938 | 24.3125 | 24.5313 | 6.599163  | 4,035,600 |

**Data Source**: [Yahoo Finance](https://finance.yahoo.com/quote/IBM/history/)

### 3.2 Data Preprocessing

#### 3.2.1 Data Normalization

In the experiment, different features have varying units and magnitudes. For example, there is a significant magnitude difference between stock prices and trading volumes, which can impact the final prediction results. Therefore, the `MinMaxScaler` method from the `sklearn.preprocessing` library is used to scale the data features to the range of 0 to 1. This not only improves model accuracy but also enhances the model’s convergence speed. The normalization formula is:

$$
x^{\prime}=\frac{x-\min (x)}{\max (x)-\min (x)}
$$

Where $x^{\prime}$ is the normalized data, $x$ is the original data, $\min (x)$ is the minimum value of the original dataset, and $\max (x)$ is the maximum value of the original dataset. After obtaining the prediction results during the experiment, the data will be denormalized for stock price prediction and model evaluation.

#### 3.2.2 Data Splitting

Here, the entire experimental datasets of Pudong Bank and IBM are input with a timestep of 60 for the recurrent kernels, and the number of input features per timestep is 5 and 6, respectively. This allows the model to input data from the previous 60 trading days to predict the closing price on the 61st day. This ensures that our dataset meets the input requirements of the three neural network models to be compared, namely the number of samples, the timestep length, and the number of input features per timestep. Subsequently, the Pudong Bank dataset is split into training, validation, and testing sets in a ratio of 2,488:311:255 after normalization. The IBM dataset is split into training, validation, and testing sets in a ratio of 6,550:364:304 after normalization. The purpose of the validation set here is to facilitate the adjustment of the model’s hyperparameters to optimize each model before comparison.

### 3.3 Model Network Structures

Through extensive experimentation, the network structures set for each model are as shown in the table below. Between layers, the default tanh and linear activation functions for recurrent neural networks are used, and Dropout with a dropout rate of 0.2 is added to prevent overfitting. Each recurrent layer in LSTM and GRU models has 50 neurons, while the BiLSTM recurrent layers have 100 neurons. Each LSTM, GRU, and BiLSTM model comprises four recurrent layers and one fully connected layer, with a Dropout layer between each network layer.

#### Table 3.3: LSTM Network Structure for IBM

| Layer(type)         | Output Shape  | Param# |
|---------------------|---------------|--------|
| lstm_1 (LSTM)       | (None, 60, 50)| 11,400 |
| dropout_1 (Dropout) | (None, 60, 50)| 0      |
| lstm_2 (LSTM)       | (None, 60, 50)| 20,200 |
| dropout_2 (Dropout) | (None, 60, 50)| 0      |
| lstm_3 (LSTM)       | (None, 60, 50)| 20,200 |
| dropout_3 (Dropout) | (None, 60, 50)| 0      |
| lstm_4 (LSTM)       | (None, 50)    | 20,200 |
| dropout_4 (Dropout) | (None, 50)    | 0      |
| dense_1 (Dense)     | (None, 1)     | 51     |

**Total params**: 72,051  
**Trainable params**: 72,051  
**Non-trainable params**: 0

---

#### Table 3.4: GRU Network Structure for IBM

| Layer(type)         | Output Shape  | Param# |
|---------------------|---------------|--------|
| gru_1 (GRU)         | (None, 60, 50)| 8,550  |
| dropout_1 (Dropout) | (None, 60, 50)| 0      |
| gru_2 (GRU)         | (None, 60, 50)| 15,150 |
| dropout_2 (Dropout) | (None, 60, 50)| 0      |
| gru_3 (GRU)         | (None, 60, 50)| 15,150 |
| dropout_3 (Dropout) | (None, 60, 50)| 0      |
| gru_4 (GRU)         | (None, 50)    | 15,150 |
| dropout_4 (Dropout) | (None, 50)    | 0      |
| dense_1 (Dense)     | (None, 1)     | 51     |

**Total params**: 54,051  
**Trainable params**: 54,051  
**Non-trainable params**: 0

---

#### Table 3.5: BiLSTM Network Structure for IBM

| Layer(type)                   | Output Shape   | Param# |
|-------------------------------|----------------|--------|
| bidirectional_1 (Bidirectional)| (None, 100)    | 22,400 |
| dropout_1 (Dropout)           | (None, 100)    | 0      |
| dense_1 (Dense)               | (None, 256)    | 25,856 |
| dropout_2 (Dropout)           | (None, 256)    | 0      |
| dense_2 (Dense)               | (None, 64)     | 16,448 |
| dense_3 (Dense)               | (None, 1)      | 0      |

**Total params**: 66,769  
**Trainable params**: 66,769  
**Non-trainable params**: 0

### 3.4 Model Compilation and Hyperparameter Settings

After iterative tuning of hyperparameters aiming to minimize the loss function on the validation set, the following settings are chosen for the three models for Pudong Bank and IBM respectively: `epochs=100`, `batch_size=32` for Pudong Bank models and `epochs=50`, `batch_size=32` for IBM models. The optimizer used is Adaptive Moment Estimation (Adam)$^{[19]}$ with default settings in the `keras` package: `lr=0.001`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-08`, and `decay=0.0`. The loss function is Mean Squared Error (MSE).

**Parameter Descriptions:**

- `lr`: Learning rate
- `beta_1`: Exponential decay rate for the first moment estimates
- `beta_2`: Exponential decay rate for the second moment estimates
- `epsilon`: Small constant for numerical stability
- `decay`: Learning rate decay per update

### 3.5 Experimental Results and Analysis

First, a brief introduction to the evaluation metrics used for model assessment is provided. The calculation formulas are as follows:

1. **Mean Squared Error (MSE)**:

$$
MSE=\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}
$$

2. **Root Mean Squared Error (RMSE)**:

$$
RMSE=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}}
$$

3. **Mean Absolute Error (MAE)**:

$$
MAE=\frac{1}{n} \sum_{i=1}^{n}\left|Y_{i}-\hat{Y}_{i}\right|
$$

4. **\( R^2 \) (R Squared)**:

$$
\begin{gathered}
\bar{Y}=\frac{1}{n} \sum_{i=1}^{n} Y_{i} \\
R^{2}=1-\frac{\sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}}{\sum_{i=1}^{n}\left(Y_{i}-\bar{Y}\right)^{2}}
\end{gathered}
$$

Where $n$ is the number of samples, $Y$ is the actual closing price, $\hat{Y}_{i}$ is the predicted closing price, and $\bar{Y}$ is the average closing price. Lower values of MSE, RMSE, and MAE indicate more accurate models. A higher $R^{2}$ indicates a better fit of the model coefficients.

#### 3.5.1 Experimental Results for Pudong Bank

##### Table 3.6: Experimental Results for Pudong Bank

|               | LSTM     | GRU      | BiLSTM   |
|---------------|----------|----------|----------|
| **MSE**       | 0.059781 | 0.069323 | 0.056454 |
| **RMSE**      | 0.244501 | 0.263292 | 0.237601 |
| **MAE**       | 0.186541 | 0.202665 | 0.154289 |
| **R-squared** | 0.91788  | 0.896214 | 0.929643 |

Comparing the evaluation metrics of the three models, we observe that on the Pudong Bank test set, the BiLSTM model has lower MSE, RMSE, and MAE than both the LSTM and GRU models, while its $R^2$ is higher than those of LSTM and GRU models. Specifically, the RMSE comparison shows that BiLSTM has a 2.90% performance improvement over LSTM and a 10.81% performance improvement over GRU on the validation set.

#### 3.5.2 Experimental Results for IBM

##### Table 3.7: Experimental Results for IBM

|               | LSTM      | GRU       | BiLSTM    |
|---------------|-----------|-----------|-----------|
| **MSE**       | 18.01311  | 12.938584 | 11.057501 |
| **RMSE**      | 4.244186  | 3.597024  | 3.325282  |
| **MAE**       | 3.793223  | 3.069033  | 2.732075  |
| **R-squared** | 0.789453  | 0.851939  | 0.883334  |

Comparing the evaluation metrics of the three models, we find that on the IBM test set, the BiLSTM model has lower MSE, RMSE, and MAE than both the LSTM and GRU models, while its $R^2$ is higher than those of LSTM and GRU models. Specifically, the RMSE comparison shows that BiLSTM has a 27.63% performance improvement over LSTM and an 8.17% performance improvement over GRU on the validation set.

### 3.6 Chapter Summary

This chapter introduced the datasets selected for the experiments—Pudong Bank and IBM—and the chosen features. It then performed data normalization and data splitting as preprocessing steps. The chapter also detailed the network structures and hyperparameter settings of the LSTM, GRU, and BiLSTM models used in the experiments. Finally, it presented the loss function curves and a series of fitting graphs for each model. By comparing multiple evaluation metrics and fitting graphs, it was concluded that the BiLSTM model can better predict stock prices, laying the foundation for the subsequent research on the LightGBM-BiLSTM quantitative investment strategy.

---

## Chapter 4: Research on LightGBM-BiLSTM-Based Quantitative Investment Models

### 4.1 Experimental Steps

{{< figure 
    src="LightGBM_BiLSTM_Flow.png" 
    caption="Fig. 11. LightGBM-BiLSTM Diagram." 
    align="center" 
>}}

As shown in Fig. 11, this experiment first selects 50 factors from the factor library. Subsequently, each factor undergoes outlier removal, normalization, and missing value imputation steps for factor cleaning. The LightGBM model is then used for factor selection, ranking factors based on importance and selecting the top ten most important factors for cross-sectional analysis. Next, the BiLSTM model is used to combine these factors to build a multi-factor model. Finally, the quantitative investment strategy is constructed based on the obtained models and subjected to backtesting analysis.

### 4.2 Experimental Data

#### 4.2.1 Stock Data

The market data used in this paper are sourced from [Tushare](https://github.com/waditu/tushare). The specific features of the dataset are shown in Table 4.1.

##### Table 4.1: Features Included in the Dataset

| Name           | Type   | Description                                     |
|----------------|--------|-------------------------------------------------|
| ts_code        | str    | Stock code                                      |
| trade_date     | str    | Trading date                                    |
| open           | float  | Opening price                                   |
| high           | float  | Highest price                                   |
| low            | float  | Lowest price                                    |
| close          | float  | Closing price                                   |
| pre_close      | float  | Previous day's closing price                    |
| change         | float  | Price change                                    |
| pct_chg        | float  | Percentage change (unadjusted)                  |
| vol            | float  | Trading volume (in hands)                       |
| amount         | float  | Trading amount (in thousands of yuan)            |

The entire A-share daily market dataset contains 5,872,309 rows of data, representing 5,872,309 samples. As shown in Table 4.2, the A-share daily market dataset has the following 11 features: stock code (ts_code), trading date (trade_date), opening price (open), closing price (close), highest price (high), lowest price (low), previous day's closing price (pre_close), price change (change), turnover rate (turnover_rate), trading amount (amount), total market value (total_mv), and adjustment factor (adj_factor).

##### Table 4.2: Partial Display of A-share Daily Market Dataset

| ts_code     | trade_date | open  | high  | low   | close | pre_close | change | vol       | amount        |
|-------------|------------|-------|-------|-------|-------|-----------|--------|-----------|---------------|
| 600613.SH   | 20120104   | 8.20  | 8.20  | 7.84  | 7.86  | 8.16      | -0.30  | 4,762.98  | 3,854.1000    |
| 600690.SH   | 20120104   | 9.00  | 9.17  | 8.78  | 8.78  | 8.93      | -0.15  | 142,288.41| 127,992.6050   |
| 300277.SZ   | 20120104   | 22.90 | 22.98 | 20.81 | 20.88 | 22.68     | -1.80  | 12,212.39 | 26,797.1370    |
| 002403.SZ   | 20120104   | 8.87  | 8.90  | 8.40  | 8.40  | 8.84      | -0.441 | 10,331.97 | 9,013.4317     |
| 300179.SZ   | 20120104   | 19.99 | 20.32 | 19.20 | 19.50 | 19.96     | -0.46  | 1,532.31  | 3,008.0594     |
| 600000.SH   | 20120104   | 8.54  | 8.56  | 8.39  | 8.41  | 8.49      | -0.08  | 342,013.79| 290,229.5510    |
| 300282.SZ   | 20120104   | 22.90 | 23.33 | 21.02 | 21.02 | 23.35     | -2.33  | 38,408.60 | 86,216.2356    |
| 002319.SZ   | 20120104   | 9.74  | 9.95  | 9.38  | 9.41  | 9.73      | -0.32  | 4,809.74  | 4,671.4803     |
| 601991.SH   | 20120104   | 5.17  | 5.39  | 5.12  | 5.25  | 5.16      | 0.09   | 145,268.38| 76,547.7490    |
| 000780.SZ   | 20120104   | 10.42 | 10.49 | 10.00 | 10.00 | 10.30     | -0.30  | 20,362.30 | 20,830.1761     |

**[5,872,309 rows x 11 columns]**

The CSI All Share Index daily dataset contains 5,057 rows of data, representing 5,057 samples. As shown in Table 4.3, the CSI All Share daily dataset has the following 7 features: trading date (trade_date), opening price (open), highest price (high), lowest price (low), closing price (close), trading volume (volume), and previous day's closing price (pre_close).

##### Table 4.3: Partial Display of CSI All Share Daily Dataset

| trade_date | open      | high      | low       | close     | volume         | pre_close |
|------------|-----------|-----------|-----------|-----------|----------------|-----------|
| 2006-11-24 | 1564.3560 | 1579.3470 | 1549.9790 | 1576.1530 | 7.521819e+09   | 1567.0910 |
| 2006-11-27 | 1574.1130 | 1598.7440 | 1574.1130 | 1598.7440 | 7.212786e+09   | 1581.1530 |
| 2006-11-28 | 1597.7200 | 1604.7190 | 1585.3620 | 1596.8400 | 7.025637e+09   | 1598.7440 |
| 2006-11-29 | 1575.3030 | 1620.2870 | 1575.3030 | 1617.9880 | 7.250354e+09   | 1596.8400 |
| 2006-11-30 | 1621.4280 | 1657.3230 | 1621.4280 | 1657.3230 | 9.656888e+09   | 1617.9880 |
| ...        | ...       | ...       | ...       | ...       | ...            | ...       |
| 2020-11-11 | 5477.8870 | 5493.5867 | 5422.9110 | 5425.8017 | 5.604086e+10   | 5494.1042 |
| 2020-11-12 | 5439.2296 | 5454.3452 | 5413.9659 | 5435.1379 | 4.594251e+10   | 5425.8017 |
| 2020-11-13 | 5418.2953 | 5418.3523 | 5364.2031 | 5402.7702 | 4.688916e+10   | 5435.1379 |
| 2020-11-16 | 5422.3565 | 5456.7264 | 5391.9232 | 5456.7264 | 5.593672e+10   | 5402.7702 |
| 2020-11-17 | 5454.0696 | 5454.0696 | 5395.6052 | 5428.0765 | 5.857009e+10   | 5456.7264 |

**[5,057 rows x 7 columns]**

Below is partial data of the original factors. After undergoing the four factor cleaning steps—missing value imputation, outlier removal, factor normalization, and factor neutralization—the cleaned factor data is displayed in the table below.

##### Table 4.4: Original Factor Data

| trade_date | sec_code    | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
|------------|-------------|----------|----------|----------|----------|----------|----------|----------|----------|-----|
| 2005-01-04 | 600874.SH   | 0.001684 | NaN      | 9.445412 | 9.445412 | 9.445408 | -1.00    | NaN      | 12651.124023 | ... |
| 2005-01-04 | 000411.SZ   | 0.021073 | NaN      | 5.971262 | 5.971262 | 5.971313 | 0.38     | NaN      | 392.124298 | ... |
| 2005-01-04 | 000979.SZ   | 0.021207 | NaN      | 6.768918 | 6.768918 | 6.768815 | -1.45    | NaN      | 870.587585 | ... |
| 2005-01-04 | 000498.SZ   | 0.030220 | NaN      | 8.852752 | 8.852752 | 8.852755 | 0.55     | NaN      | 6994.011719 | ... |
| 2005-01-04 | 600631.SH   | 0.015699 | NaN      | 9.589897 | 9.589897 | 9.589889 | -1.70    | NaN      | 14616.806641 | ... |

##### Table 4.5: Cleaned Factor Data

| sec_code  | trade_date | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
|-----------|------------|----------|----------|----------|----------|----------|----------|----------|----------|-----|
| 000001.SZ | 2005-01-04 | -1.58653 | 0.01545  | 1.38306  | 1.38306  | 1.38306  | 0.13392  | 0.01545  | 1.38564  | ... |
| 000002.SZ | 2005-01-04 | 1.36761  | -0.44814 | 1.69728  | 1.69728  | 1.69728  | 1.04567  | -0.44814 | 1.69728  | ... |
| 000004.SZ | 2005-01-04 | 0.32966  | -1.41654 | -0.13907 | -0.13907 | -0.13907 | -0.34769 | -1.41654 | -0.13650  | ... |
| 000005.SZ | 2005-01-04 | 0.61297  | -1.13066 | 1.05339  | 1.05339  | 1.05339  | -1.20020 | -1.13066 | 1.05597  | ... |
| 000006.SZ | 2005-01-04 | -0.35542 | 1.67667  | -0.07726 | -0.07726 | -0.07726 | 1.55820  | 1.67667  | -0.07469  | ... |

### 4.3 Factor Cleaning

#### 4.3.1 Handling Missing Values and Outlier Removal

Methods for handling missing values in factors include case elimination, mean substitution, regression imputation, etc. This paper adopts the relatively simple mean substitution method to handle missing values, replacing missing data with the factor’s average value. For outlier removal, methods include median trimming, percentile-based trimming, and $3\sigma$ trimming. This paper uses the $3\sigma$ trimming method, which applies the statistical $3\sigma$ principle to transform extreme factor values more than three standard deviations away from the mean to exactly three standard deviations away from the mean. The specific calculation formula is as follows:

$$
X_i^{\prime}= \begin{cases} \bar{X}+3 \sigma & \text{if } X_i > \bar{X} + 3 \sigma \\ \bar{X}-3 \sigma & \text{if } X_i < \bar{X} - 3 \sigma \\ X_i & \text{if } \bar{X} - 3 \sigma < X_i < \bar{X} + 3 \sigma \end{cases}
$$

Where:

- $X_{i}$: Original value of the factor before processing
- $\bar{X}$: Mean of the factor series
- $\sigma$: Standard deviation of the factor series
- $X_{i}^{\prime}$: Value of the factor after outlier removal

#### 4.3.2 Factor Normalization

In this experiment, multiple factors with different dimensions are selected, and their units are not entirely consistent. To facilitate comparison and regression, normalization of factors is necessary. Common normalization methods include Min-Max normalization, Z-score normalization, and Decimal scaling normalization. This paper adopts the Z-score normalization method, standardizing data using the mean and standard deviation of the original data. The processed data follow a standard normal distribution with a mean of 0 and a standard deviation of 1, resulting in normalized values that are both positive and negative, forming a standard normal distribution curve.

The Z-score normalization formula used in this paper is as follows:

$$
\tilde{x}=\frac{x_{i}-u}{\sigma}
$$

Where:

- $x_{i}$: Original value of the factor
- $u$: Mean of the factor series
- $\sigma$: Standard deviation of the factor series
- $\tilde{x}$: Normalized factor value

#### 4.3.3 Factor Neutralization

Factor neutralization aims to eliminate the influence of other factors on the selected factor, ensuring that the stocks chosen for constructing the quantitative investment strategy are more diversified rather than concentrated in specific industries or market capitalizations. This better distributes investment portfolio risk and addresses multicollinearity issues among factors. Market capitalization and industry are the two main independent variables influencing stock returns; therefore, they must be considered during factor cleaning. In this paper's empirical study, only industry factors are included, with the market factor incorporated into the industry factors. The single-factor regression model for neutralizing factors is shown in formula (31). The residuals from the following regression model are used as the new, neutralized factor values.

$$
\tilde{r}_{j}^{t}=\sum_{s=1}^{S} X_{j s}^{t} \tilde{f}_{s}^{t} + X_{j k}^{t} \tilde{f}_{k}^{t} + \tilde{u}_{j}^{t}
$$

Where:

- $\tilde{r}_{j}^{t}$: Return of stock $j$ in period $t$
- $X_{j s}^{t}$: Exposure of stock $j$ to industry $s$ in period $t$
- $\tilde{f}_{s}^{t}$: Return of industry $s$ in period $t$
- $X_{j k}^{t}$: Exposure of stock $j$ to factor $k$ in period $t$
- $\tilde{f}_{k}^{t}$: Return of factor $k$ in period $t$
- $\tilde{u}_j^t$: A $0-1$ dummy variable, indicating whether stock $j$ belongs to industry $s$ (1) or not (0)

In this paper, the industry assignment is not proportionally split; that is, stock $j$ can belong to only one specific industry $s$, with an exposure of 1 to industry $s$ and 0 to all other industries. This paper uses the Shenwan Hongyuan industry classification standard, categorizing companies into 28 primary industries: Agriculture, Forestry, Animal Husbandry, and Fishery; Mining; Chemicals; Steel; Non-Ferrous Metals; Electronic Components; Home Appliances; Food and Beverage; Textile and Apparel; Light Manufacturing; Pharmaceutical and Biotechnology; Public Utilities; Transportation; Real Estate; Commerce and Trade; Catering and Tourism; Comprehensive; Building Materials; Architectural Decoration; Electrical Equipment; National Defense and Military Industry; Computers; Media; Telecommunications; Banking; Non-Banking Finance; Automobiles; and Machinery Equipment. The table below shows the historical market data of the Shenwan Hongyuan primary industry index as of February 5, 2021.

##### Table 4.9: Historical Market Data of Shenwan Hongyuan Primary Industry Index as of February 5, 2021

| Index Code | Index Name | Release Date    | Opening Index | Highest Index | Lowest Index | Closing Index | Trading Volume (Billion) | Trading Amount (Hundred Million Yuan) | Percentage Change (%) |
|------------|------------|------------------|---------------|---------------|--------------|---------------|-------------------------|---------------------------------------|-----------------------|
| 801010     | Agriculture, Forestry, Animal Husbandry, and Fishery | 2021/2/5 0:00  | 4,111.43  | 4,271.09  | 4,072.53  | 4,081.81  | 1.581        | 307.82       | -0.3                  |
| 801020     | Mining     | 2021/2/5 0:00  | 2,344.62  | 2,357.33  | 2,288.97  | 2,289.41  | 1.806        | 115.6        | -2.25                 |
| 801030     | Chemicals  | 2021/2/5 0:00  | 4,087.77  | 4,097.59  | 3,910.67  | 3,910.67  | 5.578        | 778.85       | -3.95                 |
| 801040     | Steel      | 2021/2/5 0:00  | 2,253.78  | 2,268.17  | 2,243.48  | 2,250.81  | 1.161        | 48.39        | -1.02                 |
| 801050     | Non-Ferrous Metals | 2021/2/5 0:00  | 4,212.10  | 4,250.59  | 4,035.99  | 4,036.74  | 4.541        | 593.92       | -4.43                 |
| 801080     | Electronic Components | 2021/2/5 0:00  | 4,694.80  | 4,694.80  | 4,561.95  | 4,561.95  | 5.267        | 850.79       | -2.78                 |
| 801110     | Home Appliances | 2021/2/5 0:00  | 10,033.82 | 10,171.26 | 9,968.93  | 10,096.83 | 0.855        | 149.18       | 0.83                  |
| 801120     | Food and Beverage | 2021/2/5 0:00  | 30,876.33 | 31,545.02 | 30,649.57 | 30,931.69 | 1.132        | 657.11       | 0.47                  |
| 801130     | Textile and Apparel | 2021/2/5 0:00  | 1,614.48  | 1,633.89  | 1,604.68  | 1,607.63  | 0.628        | 57.47        | -0.39                 |
| 801140     | Light Manufacturing | 2021/2/5 0:00  | 2,782.07  | 2,791.88  | 2,735.48  | 2,737.24  | 1.528        | 176.16       | -1.35                 |
| ...        | ...        | ...              | ...         | ...         | ...         | ...         | ...                     | ...                                   | ...                   |

**Data Source**: Shenwan Hongyuan

Below is partial data of the original factors. After undergoing the four factor cleaning steps—missing value imputation, outlier removal, factor normalization, and factor neutralization—the cleaned factor data is displayed in the table below.

##### Table 4.10: Original Factor Data

| trade_date | sec_code    | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
|------------|-------------|----------|----------|----------|----------|----------|----------|----------|----------|-----|
| 2005-01-04 | 600874.SH   | 0.001684 | NaN      | 9.445412 | 9.445412 | 9.445408 | -1.00    | NaN      | 12651.124023 | ... |
| 2005-01-04 | 000411.SZ   | 0.021073 | NaN      | 5.971262 | 5.971262 | 5.971313 | 0.38     | NaN      | 392.124298 | ... |
| 2005-01-04 | 000979.SZ   | 0.021207 | NaN      | 6.768918 | 6.768918 | 6.768815 | -1.45    | NaN      | 870.587585 | ... |
| 2005-01-04 | 000498.SZ   | 0.030220 | NaN      | 8.852752 | 8.852752 | 8.852755 | 0.55     | NaN      | 6994.011719 | ... |
| 2005-01-04 | 600631.SH   | 0.015699 | NaN      | 9.589897 | 9.589897 | 9.589889 | -1.70    | NaN      | 14616.806641 | ... |

##### Table 4.11: Cleaned Factor Data

| sec_code  | trade_date | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
|-----------|------------|----------|----------|----------|----------|----------|----------|----------|----------|-----|
| 000001.SZ | 2005-01-04 | -1.58653 | 0.01545  | 1.38306  | 1.38306  | 1.38306  | 0.13392  | 0.01545  | 1.38564  | ... |
| 000002.SZ | 2005-01-04 | 1.36761  | -0.44814 | 1.69728  | 1.69728  | 1.69728  | 1.04567  | -0.44814 | 1.69728  | ... |
| 000004.SZ | 2005-01-04 | 0.32966  | -1.41654 | -0.13907 | -0.13907 | -0.13907 | -0.34769 | -1.41654 | -0.13650  | ... |
| 000005.SZ | 2005-01-04 | 0.61297  | -1.13066 | 1.05339  | 1.05339  | 1.05339  | -1.20020 | -1.13066 | 1.05597  | ... |
| 000006.SZ | 2005-01-04 | -0.35542 | 1.67667  | -0.07726 | -0.07726 | -0.07726 | 1.55820  | 1.67667  | -0.07469  | ... |

### 4.4 Factor Selection Based on LightGBM

[Freeman (2001)](https://www.jstor.org/stable/2699986)$^{[20]}$ proposed Gradient Boosting Decision Trees (GBDT), an iterative regression decision tree method. The main idea is to iteratively add weak classifiers (usually decision trees) to optimize the model, minimizing the loss function. GBDT can be expressed as:

$$
\hat{y} = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})
$$

Where:

- \( M \) is the number of iterations
- \( \gamma_m \) is the weight of the \( m \)-th weak classifier
- \( h_m(\mathbf{x}) \) is the \( m \)-th decision tree model

The training process of GBDT minimizes the loss function by sequentially fitting in the direction of the negative gradient, with the specific update formula as:

$$
\gamma_m = \arg\min_\gamma \sum_{i=1}^{N} L\left(y_i, \hat{y}_{i}^{(m-1)} + \gamma h_m(\mathbf{x}_i)\right)
$$

Where \( L \) is the loss function, \( y_i \) is the true value, and \( \hat{y}_{i}^{(m-1)} \) is the prediction after the \( m-1 \)-th iteration.

[Light Gradient Boosting Machine (LightGBM)](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)$^{[21]}$ is an efficient implementation framework of the GBDT algorithm, initially developed by Microsoft as a free and open-source distributed gradient boosting framework. LightGBM is based on decision tree algorithms and is widely used in ranking, classification, and other machine learning tasks. Its development focuses on performance and scalability, with major advantages including highly efficient parallel training, faster training speed, lower memory consumption, better accuracy, support for distributed computing, and fast processing of massive data$^{[22]}$.

The core algorithm of LightGBM is based on the following optimization objective:

$$
L = \sum_{i=1}^{N} l(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(h_m)
$$

Where \( l \) is the loss function, \( \Omega \) is the regularization term controlling model complexity, typically expressed as:

$$
\Omega(h_m) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

Here, \( T \) is the number of leaves in the tree, \( w_j \) is the weight of the \( j \)-th leaf, and \( \gamma \) and \( \lambda \) are regularization parameters.

LightGBM employs techniques such as Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to significantly enhance training efficiency and model performance.

In this study, the loss function used during training is Mean Squared Error (MSE), defined as:

$$
L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where \( y \) is the true return, \( \hat{y} \) is the predicted return, and \( N \) is the number of samples.

The specific algorithm process of this subsection is as follows:

1. **Data Preparation**: Use one year's worth of data for each stock in the A-share market, consisting of 50 factor data points and the historical return for the next month as features.
2. **Model Training**: Utilize Grid Search to optimize the LightGBM model’s hyperparameters and train the model to predict the next month's returns. The model training process is shown in Fig. 12.

   $$
   \text{Parameter Optimization:} \quad \theta^* = \arg\min_\theta \sum_{i=1}^{N} L(y_i, \hat{y}_i(\theta))
   $$

   Where \( \theta \) represents the set of model hyperparameters, and \( \theta^* \) is the optimal set.

3. **Factor Importance Calculation**: Use LightGBM's `feature_importances_` method to calculate the feature importance of each factor. Feature importance is primarily measured using two indicators:
   - **Split**: The number of times the feature is used for splitting across all trees.
   - **Gain**: The total gain brought by the feature across all splits (i.e., the reduction in the loss function).

   The feature importance of factors can be expressed as:

   $$
   \text{Importance}_{\text{split}}(f) = \sum_{m=1}^{M} \sum_{j=1}^{T_m} \mathbb{I}(f \text{ used in the split of the } j\text{-th leaf node of the } m\text{-th tree})
   $$

   $$
   \text{Importance}_{\text{gain}}(f) = \sum_{m=1}^{M} \sum_{j=1}^{T_m} \Delta L_{m,j} \cdot \mathbb{I}(f \text{ used in the split of the } j\text{-th leaf node of the } m\text{-th tree})
   $$

   Where \( \mathbb{I} \) is the indicator function, and \( \Delta L_{m,j} \) is the loss reduction brought by factor \( f \) in the split of the \( j \)-th leaf node of the \( m \)-th tree.

4. **Factor Selection**: Rank the factors based on their calculated importance and select the top ten most important factors for cross-sectional analysis, as shown in Table 4.9.

##### Table 4.9: Partial Factor Importance Ranking

| Importance | Feature Name | Trade Date  |
|------------|--------------|-------------|
| 35         | factor_35    | 2010-08-11  |
| 27         | factor_27    | 2010-08-11  |
| 33         | factor_33    | 2010-08-11  |
| 20         | factor_20    | 2010-08-11  |
| 24         | factor_24    | 2010-08-11  |
| 45         | factor_45    | 2010-08-11  |
| 37         | factor_37    | 2010-08-11  |
| 49         | factor_49    | 2010-08-11  |
| 19         | factor_19    | 2010-08-11  |
| 47         | factor_47    | 2010-08-11  |
| 22         | factor_22    | 2010-09-09  |
| 20         | factor_20    | 2010-09-09  |
| 30         | factor_30    | 2010-09-09  |
| 24         | factor_24    | 2010-09-09  |


5. Code Implementation Snippet: The following code snippet demonstrates part of the training process used for factor selection.

```python
def feature_choice(
        self,
        days=21,
        is_local=False
):
    if is_local:
        feature_info = pd.read_hdf(os.path.join(RESULTS, Feature_Info + '.h5'))
    else:
        factors = self.get_env().query_data(Factors_Data)
        factors = factors[
            factors[COM_DATE] >= '2010-01-01'
        ]
        trade_list = list(set(factors[COM_DATE]))
        trade_list.sort()
        if len(trade_list) % days == 0:
            n = int(len(trade_list) / days) - 7
        else:
            n = int(len(trade_list) / days) - 6
        feature_info = pd.DataFrame()
        begin_index = 147
        feature = list(factors.columns)
        feature.remove(COM_SEC)
        feature.remove(COM_DATE)
        feature.remove(Ret)
        for i in range(n):
            end_date = days * i + begin_index - 21
            begin_date = days * i
            trade_date = days * i + begin_index
            print(trade_list[trade_date])
            train_data = factors[
                (factors[COM_DATE] <= trade_list[end_date]) &
                (factors[COM_DATE] >= trade_list[begin_date])
            ]
            model = lgb.LGBMRegressor()
            model.fit(train_data[feature], train_data[Ret])
            feature_info_cell = pd.DataFrame(columns=Info_Fields)
            feature_info_cell[Importance] = model.feature_importances_
            feature_info_cell[Feature_Name] = model.feature_name_
            feature_info_cell = feature_info_cell.sort_values(by=Importance).tail(10)
            feature_info_cell[COM_DATE] = trade_list[trade_date]
            feature_info = pd.concat(
                [feature_info, feature_info_cell],
                axis=0
            )
        h = pd.HDFStore(os.path.join(RESULTS, Feature_Info + '.h5'), 'w')
        h['data'] = feature_info
        h.close()
    self.get_env().add_data(feature_info, Feature_Info)
    pass
```

Through the above process, LightGBM efficiently screens out factors that have the most significant impact on predicting future returns, thereby enhancing the model’s predictive capability and interpretability.

### 4.5 Factor Combination Based on BiLSTM

This subsection uses BiLSTM for factor combination. The specific principles of BiLSTM have been introduced in Chapter 2, so they will not be reiterated here. Below, we describe the specific network structure used. After extensive experimentation, the BiLSTM network structure is set as shown in Table 4.10 and Fig. 12. Between layers, the default tanh and linear activation functions for recurrent neural networks are used. To prevent overfitting, Dropout with a dropout rate of 0.01 is added. However, if the dropout rate is too high, it can lead to underfitting. The BiLSTM recurrent layer has 100 neurons, and the network comprises one BiLSTM layer and three fully connected layers, with a Dropout layer between the BiLSTM layer and the first fully connected layer.

#### Table 4.10: BiLSTM Network Structure

| Layer(type)                   | Output Shape   | Param# |
|-------------------------------|----------------|--------|
| bidirectional_1 (Bidirectional)| (None, 100)    | 22,400 |
| dropout_1 (Dropout)           | (None, 100)    | 0      |
| dense_1 (Dense)               | (None, 256)    | 25,856 |
| dropout_2 (Dropout)           | (None, 256)    | 0      |
| dense_2 (Dense)               | (None, 64)     | 16,448 |
| dense_3 (Dense)               | (None, 1)      | 0      |

**Total params**: 66,769  
**Trainable params**: 66,769  
**Non-trainable params**: 0

Due to the large volume of training data in this experiment, `epochs=400` and `batch_size=1024` are selected for the BiLSTM model. The loss function used is Mean Squared Error (MSE), and the optimizer chosen is Stochastic Gradient Descent (SGD). Compared to Gradient Descent (GD), SGD has the advantages of more effectively utilizing information in redundant data, excellent performance in early iterations, and suitability for handling large sample datasets$^{[23]}$. Given the large training data volume, using SGD allows for faster training as it updates with one sample at a time, significantly reducing training time. The default settings in the `keras` package are used: `lr=0.01`, `momentum=0.0`, `decay=0.0`, and `nesterov=False`.

**Parameter Descriptions:**

- `lr`: Learning rate
- `momentum`: Momentum parameter
- `decay`: Learning rate decay after each update
- `nesterov`: Whether to use Nesterov momentum

The specific algorithm process of this subsection is as follows:

1. **Data Usage**: Use one year's worth of data for each stock in the A-share market, consisting of 10 factors selected by LightGBM and the historical return for the next month as features.
2. **Model Training**: Use the BiLSTM model to train with the selected factors to predict the next month's returns, as shown in Fig. 12.

{{< figure 
    src="Rolling_Window.png"
    caption="Fig. 12. Rolling Window"
    align="center" 
>}}

3. **Prediction**: Apply the trained BiLSTM model to the out-of-sample data of one month to obtain the expected returns for each stock in the next month. The predicted returns are shown in Table 4.11.

##### Table 4.11: Partial Display of Predicted Returns for Stocks

| sec_code  | trade_date | y_hat      |
|-----------|------------|------------|
| 000001.SZ | 2011/5/26  | 0.0424621  |
| 000002.SZ | 2011/5/26  | -0.1632174 |
| 000004.SZ | 2011/5/26  | -0.0642319 |
| 000005.SZ | 2011/5/26  | 0.08154649 |
| 000006.SZ | 2011/5/26  | 0.00093213 |
| 000007.SZ | 2011/5/26  | -0.073218  |
| 000008.SZ | 2011/5/26  | -0.0464256 |
| 000009.SZ | 2011/5/26  | -0.091549  |
| 000010.SZ | 2011/5/26  | 0.08154649 |
| 000011.SZ | 2011/5/26  | -0.1219943 |
| 000012.SZ | 2011/5/26  | -0.1448984 |
| 000014.SZ | 2011/5/26  | 0.09038845 |
| 000016.SZ | 2011/5/26  | -0.11225   |

4. Code Implementation Snippet: The following code snippet demonstrates part of the training process used to construct a BiLSTM network.

```python
def build_net_blstm(self):
    model = ks.Sequential()
    model.add(
        ks.layers.Bidirectional(ks.layers.LSTM(
            50
        ),input_shape=(11,10))
    )
    model.add(
        ks.layers.Dropout(0.01) 
    )
    model.add(ks.layers.Dense(256))
    model.add(
        ks.layers.Dropout(0.01)
    )
    model.add(ks.layers.Dense(64))
    model.add(ks.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mse')
    model.summary()
    self.set_model(model)  
```


### 4.6 Quantitative Strategy and Strategy Backtesting

#### 4.6.1 Backtesting Metrics

Below are common backtesting metrics used to evaluate the strategy. The metrics include Total Rate of Return, Annualized Rate of Return, Annualized Volatility, Sharpe Ratio, Maximum Drawdown (MDD), Annualized Turnover Rate, and Annualized Transaction Cost Rate, assuming 252 trading days in a year and a risk-free rate of 3.5%. The transaction fee is assumed to be 0.2% per trade.

1. **Total Rate of Return**: Indicates the cumulative return of the strategy. The higher the cumulative return, the better the strategy's performance. The formula is as follows:

$$
\text{Total Rate of Return} = r_{p} = \frac{P_{1} - P_{0}}{P_{0}}
$$

Where:
- $P_{1}$: Total value of stocks and cash at the end
- $P_{0}$: Total value of stocks and cash at the beginning

2. **Annualized Rate of Return**: Converts the total cumulative return into a geometric average return on an annual basis. The higher the annualized return, the better the strategy. The formula is as follows:

$$
\text{Annualized Rate of Return} = R_{p} = \left(1 + r_{p}\right)^{\frac{252}{t}} - 1
$$

Where:
- $r_{p}$: Total rate of return
- $t$: Number of trading days the strategy is executed

3. **Annualized Volatility**: Defined as the standard deviation of the logarithm of the asset’s annual return rates. It measures the risk of the strategy, with higher volatility indicating higher risk. The formula is as follows:

$$
\begin{aligned}
\text{Annualized volatility} = \sigma_{p} &= \sqrt{\frac{252}{t-1} \sum_{i=1}^{t}\left(r_{d} - \bar{r}_{d}\right)^{2}} \\
\bar{r}_{d} &= \frac{1}{t} \sum_{i=1}^{t} r_{d_{i}}
\end{aligned}
$$

Where:
- $r_{d_{i}}$: Daily return on the $i$-th day
- $\bar{r}_{d}$: Daily average return
- $t$: Number of trading days the strategy is executed

4. **Sharpe Ratio**: Proposed by [Sharpe (1966)](https://doi.org/10.2307/2328485)$^{[24]}$, it represents the excess return per unit of risk. Here, the annualized Sharpe ratio is calculated as follows:

$$
S = \frac{R_{p} - R_{f}}{\sigma_{p}}
$$

Where:
- $R_{p}$: Annualized rate of return
- $R_{f}$: Risk-free rate
- $\sigma_{p}$: Annualized volatility

5. **Maximum Drawdown (MDD)**: Represents the maximum decline in the total value of stocks and cash during the strategy’s execution period. It measures the most extreme potential loss scenario of the strategy.

$$
MDD = \frac{\max \left(V_{x} - V_{y}\right)}{V_{x}}
$$

Where:
- $V_{x}$ and $V_{y}$ are the total values of stocks and cash on day $x$ and day $y$, respectively, with $x < y$.

6. **Annualized Turnover Rate**: Measures the frequency of buying and selling stocks within the investment portfolio. A higher turnover rate indicates more frequent rebalancing and higher transaction costs.

$$
\text{Change} = \frac{N \times 252}{t}
$$

Where:
- $t$: Number of trading days the strategy is executed
- $N$: Total number of buy and sell operations

7. **Annualized Transaction Cost Rate**: Measures the transaction costs of the investment portfolio strategy, with higher values indicating higher transaction costs.

$$
c = \left(1 + \text{commission}\right)^{\text{change}} - 1
$$

Where:
- Change: Annualized turnover rate
- Commission: Transaction fee rate

#### 4.6.2 Strategy and Backtesting Results

This paper adopts a quantitative trading strategy that rebalances the portfolio every month (i.e., a rebalancing period of 28 trading days). Each rebalance involves equally weighting the top 25 stocks with the highest expected returns predicted by BiLSTM and selling the previously held stocks. The backtesting period and rules are as follows:

1. **Backtesting Period**: From January 2012 to October 2020.
2. **Backtesting Stock Pool**: All A-share stocks, excluding Special Treatment (ST) stocks.
3. **Transaction Fees**: A trading commission of 0.2% is paid to brokers upon buying and selling, with a minimum commission of 5 yuan per trade if the calculated fee is less than 5 yuan.
4. **Trading Rules**: Stocks hitting the daily limit-up cannot be bought, and stocks hitting the daily limit-down cannot be sold on the same day.

##### Table 4.12: Strategy Backtesting Results

|                 | Total Rate of Return | Annualized Rate of Return | Annualized Volatility | Sharpe Ratio | Maximum Drawdown | Annualized Turnover Rate | Annualized Transaction Cost Rate |
|-----------------|-----------------------|---------------------------|-----------------------|--------------|-------------------|--------------------------|-----------------------------------|
| **Strategy**    | 701.00%               | 29.18%                    | 33.44%                | 0.77         | 51.10%            | 51.10%                   | 11.35%                            |
| **Benchmark**   | 110.40%               | 9.70%                     | 26.01%                | 0.24         | 58.49%            | 58.49%                   | 0.00%                             |

{{< figure 
    src="res.png" 
    caption="Fig. 22. Net Profit Curve" 
    align="center" 
>}}

The backtesting results are shown in Table 4.12 and Fig. 22. The strategy employed is the LightGBM-BiLSTM quantitative strategy introduced in this chapter. The benchmark used is the CSI All Share Index (000985). From the results, the strategy achieves a cumulative return of 701.00%, significantly higher than the benchmark’s 110.40%. The annualized return is 29.18%, far exceeding the benchmark’s 9.70%. The Sharpe ratio is 0.77, which is higher than the benchmark’s 0.24. These three backtesting metrics indicate that the LightGBM-BiLSTM quantitative strategy indeed provides greater returns to investors, demonstrating the effectiveness of using deep learning to construct quantitative investment strategies. The strategy’s annualized volatility is 33.44%, higher than the benchmark’s 26.01%, and the maximum drawdown is 51.10%, lower than the benchmark’s 58.49%. These two metrics indicate that the LightGBM-BiLSTM quantitative strategy carries certain risks, particularly in resisting systemic risk shocks. The annualized turnover rate is 11.35%, and the annualized transaction cost rate is 2.29%, indicating that the strategy is not a high-frequency trading strategy with relatively low transaction costs. The return curve shows that the LightGBM-BiLSTM quantitative strategy’s returns were similar to the benchmark in the first two years without notable advantages. However, starting around April 2015, the strategy’s returns clearly outperformed the benchmark. Overall, the LightGBM-BiLSTM quantitative strategy delivers substantial returns but still carries certain risks.

## Chapter 5: Conclusion and Outlook

### 5.1 Conclusion

This paper first introduced the research background and significance of deep learning-based stock price prediction and quantitative investment strategy research. It then reviewed the current domestic and international research status of stock price prediction and quantitative investment strategies, outlined the innovations of this paper, and presented the research framework. Next, the theoretical foundations chapter briefly introduced the deep learning models and the development history of quantitative investment used in this paper, with a focus on the basic structures, principles, and characteristics of LSTM, GRU, and BiLSTM models.

Subsequently, using daily data from Pudong Bank and IBM, the paper conducted a series of data processing steps and feature extraction for data preprocessing. It then detailed the network structures and hyperparameter settings of the LSTM, GRU, and BiLSTM models used in the experiments. The study used LSTM, GRU, and BiLSTM to predict the closing prices of the two stocks and compared the models’ performance. The experimental results showed that the BiLSTM model achieved higher prediction accuracy for both stocks.

Finally, to further demonstrate BiLSTM’s application value in finance, this paper constructed a LightGBM-BiLSTM-based quantitative investment model. It selected and cleaned multiple factors from the entire A-share market data, employed LightGBM for factor selection, and used BiLSTM for factor combination. The paper then constructed a quantitative investment strategy based on the model and compared it against the CSI All Share Index using metrics such as cumulative return, annualized return, annualized volatility, and Sharpe ratio. The comparison showed that the LightGBM-BiLSTM quantitative investment model achieved better returns, indicating the effectiveness of using deep learning to build quantitative investment strategies.

### 5.2 Outlook

Although this paper compared the performance of LSTM, GRU, and BiLSTM models in predicting stock closing prices and achieved certain results with the LightGBM-BiLSTM quantitative investment strategy, there are still some shortcomings. Based on the research outcomes of this paper, the following areas can be further explored and improved:

1. **Diverse Prediction Targets**: This paper focuses on predicting stock closing prices, which, while intuitive, is challenging and less interpretable under the Random Walk Hypothesis (RWH) proposed by [Bachelier (1900)](http://www.numdam.org/item/ASENS_1900_3_17__21_0/)$^{[26]}$, which posits that stock prices follow a random walk and are unpredictable. Although behavioral economists have shown that this hypothesis is not entirely correct, it indicates that simply predicting stock closing prices is difficult and less interpretable$^{[27][28]}$. Therefore, future research can explore predicting stock volatility, stock price movements (up or down), and stock returns.

2. **Diverse Model Comparisons**: This paper compared LSTM, GRU, and BiLSTM models in predicting stock prices, demonstrating that BiLSTM achieves better accuracy. However, it lacks comparisons with other diverse models. Future research can delve into comparing single or hybrid models like ARIMA, Convolutional Neural Networks (CNN), Deep Neural Networks (DNN), CNN-LSTM, Transformer, and TimeGPT.

3. **Diverse Factor Selection**: The factors used in constructing the quantitative investment strategy in this paper are primarily technical price and volume factors, with a limited variety. Future research can incorporate financial factors, sentiment factors, growth factors, and other types of factors to enhance the strategy’s performance. Additionally, incorporating timing strategies—such as increasing positions when the market is predicted to rise and decreasing positions when the market is predicted to fall—can capture beta ($\beta$) returns.

4. **Investment Portfolio Optimization**: The factor combination process in this paper remains imperfect. Future research can utilize quadratic programming methods to optimize the investment portfolio.

5. **High-Frequency Trading Strategy Research**: The quantitative investment strategy in this paper adopts a low-frequency trading approach. Future research can leverage tick data to explore high-frequency and ultra-high-frequency trading strategies.

## References

[1] White, H. [“Economic prediction using neural networks: The case of IBM daily stock returns.”](https://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf) *Proc. of ICNN*. 1988, 2: 451-458.  

[2] Kimoto, T., Asakawa, K., Yoda, M., et al. [“Stock market prediction system with modular neural networks.”](https://web.ist.utl.pt/adriano.simoes/tese/referencias/Papers%20-%20Adriano/NN.pdf) *Proc. of 1990 IJCNN International Joint Conference on Neural Networks*. IEEE, 1990: 1-6.  

[3] Zhang, G. P. [“Time series forecasting using a hybrid ARIMA and neural network model.”](https://dl.icdst.org/pdfs/files/2c442c738bd6bc178e715f400bec5d5f.pdf) *Neurocomputing*. 2003, 50: 159-175.  

[4] Akita, R., Yoshihara, A., Matsubara, T., et al. [“Deep learning for stock prediction using numerical and textual information.”](https://ieeexplore.ieee.org/document/7550882) *Proc. of 2016 IEEE/ACIS 15th International Conference on Computer and Information Science (ICIS)*. IEEE, 2016: 1-6.  

[5] 宮崎邦洋, 松尾豊. [“Deep Learning を用いた株価予測の分析.”](https://www.ai-gakkai.or.jp/jsai2017/webprogram/2017/pdf/1112.pdf) *人工知能学会全国大会論文集 第31回全国大会*. 一般社団法人 人工知能学会, 2017: 2D3OS19a3-2D3OS19a3.  

[6] Kim, T., Kim, H. Y. [“Forecasting stock prices with a feature fusion LSTM-CNN model using different representations of the same data.”](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0212320&type=printable) *PLoS ONE*. 2019, 14(2): e0212320.  

[7] Hochreiter, S., Schmidhuber, J. [“Long short-term memory.”](https://www.bioinf.jku.at/publications/older/2604.pdf) *Neural Computation*. 1997, 9(8): 1735-1780.  

[8] Cho, K., Van Merriënboer, B., Gulcehre, C., et al. [“Learning phrase representations using RNN encoder-decoder for statistical machine translation.”](https://arxiv.org/abs/1406.1078) *arXiv preprint arXiv:1406.1078*. 2014.  

[9] Chung, J., Gulcehre, C., Cho, K. H., et al. [“Empirical evaluation of gated recurrent neural networks on sequence modeling.”](https://arxiv.org/abs/1412.3555) *arXiv preprint arXiv:1412.3555*. 2014.  

[10] Gruber, N., Jockisch, A. [“Are GRU cells more specific and LSTM cells more sensitive in motive classification of text?”](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00040/full) *Frontiers in Artificial Intelligence*. 2020, 3(40): 1-6.  

[11] Markowitz, H. [“Portfolio Selection.”](https://www.jstor.org/stable/2975974) *The Journal of Finance*. 1952, 7(1): 77-91. doi:10.2307/2975974.  

[12] Merton, R. C. [“An analytic derivation of the efficient portfolio frontier.”](https://www.jstor.org/stable/2329621) *Journal of Financial and Quantitative Analysis*. 1972: 1851-1872.  

[13] Sharpe, W. F. [“Capital asset prices: A theory of market equilibrium under conditions of risk.”](https://www.jstor.org/stable/2977928) *The Journal of Finance*. 1964, 19(3): 425-442.  

[14] Lintner, J. [“The Valuation of Risk Assets and the Selection of Risky Investments in Stock Portfolios and Capital Budgets.”](https://www.jstor.org/stable/1924119) *Review of Economics and Statistics*. 1965, 47(1): 13-37.  

[15] Mossin, J. [“Equilibrium in a capital asset market.”](https://www.jstor.org/stable/1910098) *Econometrica: Journal of the Econometric Society*. 1966: 768-783.  

[16] Ross, S. A. [“The arbitrage theory of capital asset pricing.”](https://www.top1000funds.com/wp-content/uploads/2014/05/The-Arbitrage-Theory-of-Capital-Asset-Pricing.pdf) *Journal of Economic Theory*. 1976, 13(3): 341-60.  

[17] Fama, E. F., French, K. R. [“Common risk factors in the returns on stocks and bonds.”](https://www.bauer.uh.edu/rsusmel/phd/Fama-French_JFE93.pdf) *Journal of Financial Economics*. 1993, 33(1): 3-56.  

[18] Fama, E. F., French, K. R. [“A five-factor asset pricing model.”](https://tevgeniou.github.io/EquityRiskFactors/bibliography/FiveFactor.pdf) *Journal of Financial Economics*. 2015, 116(1): 1-22.  

[19] Kingma, D. P., Ba, J. [“Adam: A method for stochastic optimization.”](https://arxiv.org/abs/1412.6980) *arXiv preprint arXiv:1412.6980*. 2014.  

[20] Friedman, J. H. [“Greedy function approximation: A gradient boosting machine.”](https://www.jstor.org/stable/2699986) *Annals of Statistics*. 2001: 1189-1232.  

[21] Kopitar, L., Kocbek, P., Cilar, L., et al. [“Early detection of type 2 diabetes mellitus using machine learning-based prediction models.”](https://www.nature.com/articles/s41598-020-68771-z) *Scientific Reports*. 2020, 10(1): 1-12.  

[22] Ke, G., Meng, Q., Finley, T., et al. [“Lightgbm: A highly efficient gradient boosting decision tree.”](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) *Advances in Neural Information Processing Systems*. 2017, 30: 3146-3154.  

[23] Bottou, L., Curtis, F. E., Nocedal, J. [“Optimization methods for large-scale machine learning.”](https://coral.ise.lehigh.edu/frankecurtis/files/papers/BottCurtNoce18.pdf) *SIAM Review*. 2018, 60(2): 223-311.  

[24] Sharpe, W. F. [“Mutual fund performance.”](https://finance.martinsewell.com/fund-performance/Sharpe1966.pdf) *The Journal of Business*. 1966, 39(1): 119-138.  

[25] Sharpe, W. F. [“The sharpe ratio.”](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm) *Journal of Portfolio Management*. 1994, 21(1): 49-58.  

[26] Bachelier, L. [“Théorie de la spéculation.”](http://www.numdam.org/item/ASENS_1900_3_17__21_0/) *Annales Scientifiques de l'École Normale Supérieure*. 1900, 17: 21-86.  

[27] Fromlet, H. [“Behavioral finance-theory and practical application: Systematic analysis of departures from the homo oeconomicus paradigm are essential for realistic financial research and analysis.”](https://www.jstor.org/stable/23488166) *Business Economics*. 2001: 63-69.  

[28] Lo, A. W. [“The adaptive markets hypothesis.”](https://www.pm-research.com/content/iijpormgmt/30/5/15) *The Journal of Portfolio Management*. 2004, 30(5): 15-29.  

### Reference Blogs

- Colah's Blog. (2015, August 27). [*Understanding LSTM Networks.*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


## Citation

> **Citation**: To reproduce or cite the content of this article, please acknowledge the original author and source.

**Cited as:**

> Yue Shui. (Apr 2021). Research on Stock Price Prediction and Quantitative Strategies Based on Deep Learning.  
https://syhya.github.io/posts/2021-04-21-deep-learning-stock-prediction/

Or

```bibtex
@article{syhya2021stockprediction,
  title   = "Research on Stock Price Prediction and Quantitative Strategies Based on Deep Learning",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2021",
  month   = "Apr",
  url     = "https://syhya.github.io/posts/2021-04-21-deep-learning-stock-prediction/"
}
