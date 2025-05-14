---
title: "Stock Price Prediction and Quantitative Strategy Based on Deep Learning"
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

The stock market is a crucial component of the financial market. In recent years, with its vigorous development, research on stock price prediction and quantitative investment strategies has attracted scholars from various fields.  With the advancement of Artificial Intelligence (AI) and Machine Learning (ML) in recent years, researchers have shifted from traditional statistical models to AI algorithms. Particularly after the deep learning boom, neural networks have achieved remarkable results in stock price prediction and quantitative investment strategy research. The objective of deep learning is to learn multi-level features, constructing abstract high-level features by combining low-level ones, thereby mining the distributed feature representations of data. This approach enables complex nonlinear modeling to accomplish prediction tasks. Recurrent Neural Networks (RNNs) have been widely applied to sequential data, such as natural language and speech. Daily stock prices and trading information are sequential data, leading many researchers to use RNNs for stock price prediction. However, basic RNNs suffer from gradient vanishing issues when the number of layers is excessive. The advent of Long Short-Term Memory (LSTM) networks addressed this problem, followed by variants such as Gated Recurrent Units (GRUs), Peephole LSTMs, and Bidirectional LSTMs (BiLSTMs). Traditional stock prediction models often overlook temporal factors or only consider unidirectional temporal relationships. Therefore, this paper employs the BiLSTM model for stock price prediction. From a model principle perspective, the BiLSTM model fully leverages the contextual relationships in both forward and backward temporal directions of time series data. It also avoids gradient vanishing and explosion problems in long sequences, enabling better learning of information with long-term dependencies.

The first part of this paper's experiments utilizes stock data from China's Shanghai Pudong Development Bank and the US's IBM to establish stock prediction models using LSTM, GRU, and BiLSTM. By comparing the prediction results of these three deep learning models, it is found that the BiLSTM model outperforms the others for both datasets, demonstrating better prediction accuracy. The second part uses A-share market-wide stock data and first employs a LightGBM model to screen 50 factors, selecting the top 10 most important factors. Subsequently, a BiLSTM model is used to select and combine these factors to establish a quantitative investment strategy. Empirical analysis and backtesting of this strategy reveal that it outperforms the market benchmark index, indicating the practical application value of the BiLSTM model in stock price prediction and quantitative investment.

**Keywords**: Quantitative Investment; Deep Learning; Neural Network Model; Multi-Factor Stock Selection; BiLSTM

## Chapter 1 Introduction

### 1.1 Research Background and Significance

#### 1.1.1 Research Background

Emerging in the 1970s, quantitative investment gradually entered the vision of investors, initiating a new revolution that changed the landscape of portfolio management previously dominated by passive management and the efficient market hypothesis. The efficient market hypothesis posits that under market efficiency, stock prices reflect all market information, and no excess returns exist. Passive investment management, based on the belief that markets are efficient, focuses more on asset classes, with the most common approach being purchasing index funds and tracking published index performance. In contrast, active investment management relies primarily on investors' subjective judgments of the market and individual stocks. By applying mathematical models to the financial domain and using available public data to evaluate stocks, active managers construct portfolios to achieve returns. Quantitative investment, through statistical processing of vast historical data to uncover investment opportunities and avoid subjective factors, has gained increasing popularity among investors. Since the rise of quantitative investment, people have gradually utilized various technologies for stock price prediction to better establish quantitative investment strategies. Early domestic and international scholars adopted statistical methods for modeling and predicting stock prices, such as exponential smoothing, multiple regression, and Autoregressive Moving Average (ARMA) models. However, due to the multitude of factors influencing the stock market and the large volume of data, stock prediction is highly challenging, and the prediction effectiveness of various statistical models has been unsatisfactory.

In recent years, the continuous development of machine learning, deep learning, and neural network technologies has provided technical support for stock price prediction and the construction of quantitative strategies. Many scholars have achieved new breakthroughs using methods like Random Forest, Neural Networks, Support Vector Machines, and Convolutional Neural Networks. The ample historical data in the stock market, coupled with diverse technological support, provides favorable conditions for stock price prediction and the construction of quantitative strategies.

#### 1.1.2 Research Significance

From the perspective of the long-term development of the national economic system and financial markets, research on stock price prediction models and quantitative investment strategies is indispensable. China started relatively late, with a less mature financial market, fewer financial instruments, and lower market efficiency. However, in recent years, the country has gradually relaxed policies and vigorously developed the financial market, providing a "breeding ground" for the development of quantitative investment. Developing quantitative investment and emerging financial technologies can offer China's financial market an opportunity for a "curve overtaking". Furthermore, the stock price index, as a crucial economic indicator, serves as a barometer for China's economic development.

From the perspective of individual and institutional investors, constructing stock price prediction models and quantitative investment strategy models enhances market efficiency. Individual investors often lack comprehensive professional knowledge, and their investment behaviors can be somewhat blind. Developing relevant models to provide references can reduce the probability of judgment errors and change the relatively disadvantaged position of individual investors in the capital market. For institutional investors, rational and objective models, combined with personal experience, improve the accuracy of decision-making and provide new directions for investment behaviors.

In summary, considering China's current development status, this paper's selection of individual stocks for stock price prediction models and A-share market-wide stocks for quantitative strategy research holds significant practical research value.

### 1.2 Literature Review

[White (1988)](https://pages.cs.wisc.edu/~dyer/cs540/handouts/deep-learning-nature2015.pdf)$^{[1]}$ used a Backpropagation (BP) neural network to predict the daily returns of IBM stock. However, due to the BP neural network model's susceptibility to gradient explosion, the model could not converge to a global minimum, thus failing to achieve accurate predictions.

[Kimoto (1990)](https://web.ist.utl.pt/adriano.simoes/tese/referencias/Papers%20-%20Adriano/NN.pdf)$^{[2]}$ developed a system for predicting the Tokyo Stock Exchange Prices Index (TOPIX) using modular neural network technology. This system not only successfully predicted TOPIX but also achieved a certain level of profitability through stock trading simulations based on the prediction results.

[G. Peter Zhang (2003)](https://dl.icdst.org/pdfs/files/2c442c738bd6bc178e715f400bec5d5f.pdf)$^{[3]}$ conducted a comparative study on the performance of Autoregressive Integrated Moving Average (ARIMA) models and Artificial Neural Network (ANN) models in time series forecasting. The results showed that ANN models significantly outperformed ARIMA models in terms of time series prediction accuracy.

[Ryo Akita (2016)](https://ieeexplore.ieee.org/document/7550882)$^{[4]}$ selected the Consumer Price Index (CPI), Price-to-Earnings ratio (P/E ratio), and various events reported in newspapers as features, and constructed a financial time series prediction model using paragraph vectors and LSTM networks. Using actual data from fifty listed companies on the Tokyo Stock Exchange, the effectiveness of this model in predicting stock opening prices was verified.

[Kunihiro Miyazaki (2017)](https://www.ai-gakkai.or.jp/jsai2017/webprogram/2017/pdf/1112.pdf)$^{[5]}$ constructed a model for predicting the rise and fall of the Topix Core 30 index and its constituent stocks by extracting daily stock chart images and 30-minute stock price data. The study compared multiple models, including Logistic Regression (LR), Random Forest (RF), Multilayer Perceptron (MLP), LSTM, CNN, PCA-CNN, and CNN-LSTM. The results indicated that LSTM had the best prediction performance, CNN performed generally, but hybrid models combining CNN and LSTM could improve prediction accuracy.

[Taewook Kim (2019)](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0212320&type=printable)$^{[6]}$ proposed an LSTM-CNN hybrid model to combine features from both stock price time series and stock price image data representations to predict the stock price of the S&P 500 index. The study showed that the LSTM-CNN model outperformed single models in stock price prediction, and this prediction had practical significance for constructing quantitative investment strategies.

### 1.3 Innovations of the Paper

This paper has the following innovations in stock price prediction:

1.  Data from both the domestic A-share market (Shanghai Pudong Development Bank) and the international US stock market (IBM) are used for research, avoiding the limitations of single-market studies. Traditional BP models have never considered temporal factors, or like LSTM models, only consider unidirectional temporal relationships. Therefore, this paper uses the BiLSTM model for stock price prediction. From a model principle perspective, the BiLSTM model fully utilizes the contextual relationships in both forward and backward temporal directions of time series data and avoids gradient vanishing and explosion problems in long sequences, enabling better learning of information with long-term dependencies. Empirical evidence, compared with LSTM and GRU models, demonstrates its ability to improve prediction accuracy.
2.  The stock price prediction model is trained using multiple stock features, including opening price, closing price, highest price, and trading volume. Compared to single-feature prediction of stock closing prices, this approach is theoretically more accurate and can better compare the prediction effectiveness of LSTM, GRU, and BiLSTM for stocks.

This paper has the following innovations in quantitative strategy research:

1.  Instead of using common market factors, this paper uses multiple price-volume factors obtained through Genetic Programming (GP) algorithms and artificial data mining. LightGBM model is used to screen 50 factors, selecting the top 10 most important factors.
2.  Traditional quantitative investment models generally use LSTM and CNN models to construct quantitative investment strategies. This paper uses A-share market-wide data and employs a BiLSTM model to select and combine factors to establish a quantitative investment strategy. Backtesting and empirical analysis of this strategy show that it outperforms the market benchmark index (CSI All Share), demonstrating the practical application value of the BiLSTM model in stock price prediction and quantitative investment.

### 1.4 Research Framework of the Paper

This paper conducts research on stock price prediction and quantitative strategies based on deep learning algorithms. The specific research framework of this paper is shown in Fig. 1:

{{< figure
    src="Research Framework.svg"
    caption="Fig. 1. Research Framework."
    align="center"
>}}

The specific research framework of this paper is as follows:

Chapter 1 is the introduction. This chapter first introduces the research significance and background of stock price prediction and quantitative strategy research. Then, it reviews the current research status, followed by an explanation of the innovations of this paper compared to existing research, and finally, a brief description of the research framework of this paper.

Chapter 2 is about related theoretical foundations. This chapter introduces the basic theories of deep learning models and quantitative stock selection involved in this research. The deep learning model section sequentially introduces four deep learning models: RNN, LSTM, GRU, and BiLSTM, with a focus on the internal structure of the LSTM model. The quantitative stock selection theory section sequentially introduces the Mean-Variance Model, Capital Asset Pricing Model, Arbitrage Pricing Theory, Multi-Factor Model, Fama-French Three-Factor Model, and Five-Factor Model. This section introduces the history of multi-factor quantitative stock selection from the development context of various financial theories and models.

Chapter 3 is a comparative study of LSTM, GRU, and BiLSTM in stock price prediction. This chapter first introduces the datasets of domestic and international stocks used in the experiment, and then performs data preprocessing steps of normalization and data partitioning. It then describes the network structures, model compilation, and hyperparameter settings of the LSTM, GRU, and BiLSTM models used in this chapter, and conducts experiments to obtain experimental results. Finally, the experimental results are analyzed, and a summary of this chapter is provided.

Chapter 4 is a study on a quantitative investment model based on LightGBM-BiLSTM. This chapter first outlines the experimental steps, and then introduces the stock data and factor data used in the experiment. Subsequently, factors are processed sequentially through missing value handling, outlier removal, factor standardization, and factor neutralization to obtain cleaned factors. Then, LightGBM and BiLSTM are used for factor selection and factor combination, respectively. Finally, a quantitative strategy is constructed based on the obtained model, and backtesting is performed on the quantitative strategy.

Chapter 5 is the conclusion and future directions. This chapter summarizes the main research content of this paper on stock price prediction and quantitative investment strategies. Based on the existing shortcomings of the current research, future research directions are proposed.

## Chapter 2 Related Theoretical Foundations

### 2.1 Deep Learning Models

#### 2.1.1 RNN

Recurrent Neural Networks (RNNs) are widely used for sequential data, such as natural language and speech. Daily stock prices and trading information are sequential data, hence many previous studies have used RNNs to predict stock prices. RNNs employ a very simple chain structure of repeating modules, such as a single tanh layer. However, basic RNNs suffer from gradient vanishing issues when the number of layers is excessive. The emergence of LSTM solved this problem. Fig. 2 is an RNN structure diagram.

{{< figure
    src="RNN.png"
    caption="Fig. 2. RNN Structure Diagram. (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))"
    align="center"
>}}

#### 2.1.2 LSTM

Long Short-Term Memory (LSTM) networks are a special kind of RNN, capable of learning long-term dependencies. They were introduced by [Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)$^{[7]}$ and have been refined and popularized by many in subsequent work. Due to their unique design structure, LSTMs are relatively insensitive to gap lengths and solve the gradient vanishing and explosion problems of traditional RNNs. Compared to traditional RNNs and other time series models like Hidden Markov Models (HMMs), LSTMs can better handle and predict important events in time series with very long intervals and delays. Therefore, LSTMs are widely used in machine translation and speech recognition.

LSTMs are explicitly designed to avoid long-term dependency problems. All recurrent neural networks have the form of a chain of repeating modules of neural networks, but LSTM improves the structure of RNN. Instead of a single neural network layer, LSTM uses a four-layer structure that interacts in a special way.

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

As shown in Fig. 3, black lines are used to represent the transmission of an output vector from one node to the input vector of another node. A neural network layer is a processing module with a $\sigma$ activation function or a tanh activation function; pointwise operation represents element-wise multiplication between vectors; vector transfer indicates the direction of information flow; concatenate and copy are represented by two black lines merging together and two black lines separating, respectively, to indicate information merging and information copying.

Below, we take LSTM as an example to explain its structure in detail.

1.  **Forget Gate**

{{< figure
    src="forget_gate.png"
    caption="Fig. 5. Forget Gate Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))"
    align="center"
    width="70%"
>}}

$$
f_{t} = \sigma\left(W_{f} \cdot \left[h_{t-1}, x_{t}\right] + b_{f}\right)
$$

**Parameter Description:**

-   $h_{t-1}$: Output from the previous time step
-   $x_{t}$: Input at the current time step
-   $\sigma$: Sigmoid activation function
-   $W_{f}$: Weight matrix for the forget gate
-   $b_{f}$: Bias vector parameter for the forget gate

The first step, as shown in Fig. 5, is to decide what information to discard from the cell state. This process is calculated by the sigmoid function to obtain the value of $f_{t}$ (the range of $f_{t}$ is between 0 and 1, where 0 means completely block, and 1 means completely pass through) to determine whether the cell state $C_{t-1}$ is passed through or partially passed through.

2.  **Input Gate**

{{< figure
    src="input_gate1.png"
    caption="Fig. 6. Input Gate Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)) "
    align="center"
    width="70%"
>}}

$$
\begin{aligned}
i_{t} &= \sigma\left(W_{i} \cdot \left[h_{t-1}, x_{t}\right] + b_{i}\right) \\
\tilde{C}_{t} &= \tanh\left(W_{C} \cdot \left[h_{t-1}, x_{t}\right] + b_{C}\right)
\end{aligned}
$$

**Parameter Description:**

-   $h_{t-1}$: Output from the previous time step
-   $x_{t}$: Input at the current time step
-   $\sigma$: Sigmoid activation function
-   $W_{i}$: Weight matrix for the input gate
-   $b_{i}$: Bias vector parameter for the input gate
-   $W_{C}$: Weight matrix for the cell state
-   $b_{C}$: Bias vector parameter for the cell state
-   $\tanh$: tanh activation function

The second step, as shown in Fig. 6, uses a sigmoid function to calculate what information we want to store in the cell state. Next, a $\tanh$ layer creates a candidate vector $\tilde{C}_{t}$, which will be added to the cell state.

{{< figure
    src="input_gate2.png"
    caption="Fig. 7. Current Cell State Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))"
    align="center"
    width="70%"
>}}

$$
C_{t} = f_{t} * C_{t-1} + i_{t} * \tilde{C}_{t}
$$

**Parameter Description:**

-   $C_{t-1}$: Cell state from the previous time step
-   $\tilde{C}_{t}$: Temporary cell state
-   $i_{t}$: Value of the input gate
-   $f_{t}$: Value of the forget gate

The third step, as shown in Fig. 7, calculates the current cell state $C_t$ by combining the effects of the forget gate and the input gate.
-   The forget gate $f_t$ weights the previous cell state $C_{t-1}$ to discard unnecessary information.
-   The input gate $i_t$ weights the candidate cell state $\tilde{C}_t$ to decide how much new information to introduce.
Finally, the two parts are added together to update and derive the current cell state $C_t$.

3.  **Output Gate**

{{< figure
    src="output_gate.png"
    caption="Fig. 8. Output Gate Calculation (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))"
    align="center"
    width="70%"
>}}

$$
\begin{aligned}
o_{t} &= \sigma\left(W_{o} \cdot \left[h_{t-1}, x_{t}\right] + b_{o}\right) \\
h_{t} &= o_{t} * \tanh\left(C_{t}\right)
\end{aligned}
$$

**Parameter Description:**

-   $o_{t}$: Value of the output gate
-   $\sigma$: Sigmoid activation function
-   $W_{o}$: Weight matrix for the output gate
-   $h_{t-1}$: Output from the previous time step
-   $x_{t}$: Input at the current time step
-   $b_{o}$: Bias vector parameter for the output gate
-   $h_{t}$: Output at the current time step
-   $\tanh$: tanh activation function
-   $C_{t}$: Cell state at the current time step

The fourth step, as shown in Fig. 8, uses a sigmoid function to calculate the value of the output gate. Finally, the cell state $C_{t}$ at this time step is processed by a tanh activation function and multiplied by the value of the output gate $o_{t}$ to obtain the output $h_{t}$ at the current time step.

#### 2.1.3 GRU

[K. Cho (2014)](https://arxiv.org/abs/1406.1078)$^{[8]}$ proposed the Gated Recurrent Unit (GRU). GRU is mainly simplified and adjusted based on LSTM, merging the original forget gate, input gate, and output gate of LSTM into an update gate and a reset gate. In addition, GRU also merges the cell state and hidden state, thereby reducing the complexity of the model while still achieving performance comparable to LSTM in some tasks.

This model can save a lot of time when the training dataset is relatively large and shows better performance on some smaller and less frequent datasets$^{[9][10]}$.

{{< figure
    src="GRU.png"
    caption="Fig. 9. GRU Structure Diagram (Image source: [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/))"
    align="center"
    width="70%"
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

-   $z_{t}$: Value of the update gate
-   $r_{t}$: Value of the reset gate
-   $W_{z}$: Weight matrix for the update gate
-   $W_{r}$: Weight matrix for the reset gate
-   $\tilde{h}_{t}$: Temporary output

#### 2.1.4 BiLSTM

Bidirectional Long Short-Term Memory (BiLSTM) networks are formed by combining a forward LSTM and a backward LSTM. The BiLSTM model fully utilizes the contextual relationships in both forward and backward temporal directions of time series data, enabling it to learn information with long-term dependencies. Compared to unidirectional LSTM, it can better consider the reverse impact of future data. Fig. 10 is a BiLSTM structure diagram.

{{< figure
    src="BiLSTM.png"
    caption="Fig. 10. BiLSTM Structure Diagram. (Image source: [Baeldung](https://www.baeldung.com/cs/bidirectional-vs-unidirectional-lstm))"
    align="center"
>}}

### 2.2 Quantitative Stock Selection Theory

#### 2.2.1 Mean-Variance Model

Quantitative stock selection strategies originated in the 1950s. [Markowitz (1952)](https://www.jstor.org/stable/2975974)$^{[11]}$ proposed the Mean-Variance Model. This model not only laid the foundation for modern portfolio theory, quantifying investment risk, but also established a specific model describing risk and expected return. It broke away from the previous situation of qualitative analysis of investment portfolios without quantitative analysis, successfully introducing mathematical models into the field of financial investment.

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

-   $\mathrm{E}\left(R_{p}\right)$ and $\mu_{p}$ are the expected return of portfolio $p$
-   $\mathrm{E}\left(R_{i}\right)$ is the expected return of asset $i$
-   $\sigma_{i}$ is the standard deviation of asset $i$
-   $\sigma_{j}$ is the standard deviation of asset $j$
-   $w_{i}$ is the proportion of asset $i$ in the portfolio
-   $\sigma_{p}^{2}$ is the variance of portfolio $p$
-   $\rho_{ij}$ is the correlation coefficient between asset $i$ and asset $j$

Using the above formulas$^{[12]}$, we can construct an investment portfolio that minimizes non-systematic risk under a certain expected rate of return.

#### 2.2.2 Capital Asset Pricing Model

[William Sharpe (1964)](https://www.jstor.org/stable/2977928)$^{[13]}$, [John Lintner (1965)](https://www.jstor.org/stable/1924119)$^{[14]}$, and [Jan Mossin (1966)](https://www.jstor.org/stable/1910098)$^{[15]}$ proposed the Capital Asset Pricing Model (CAPM). This model posits that the expected return of an asset is related to its risk measure, the $\beta$ value. Through a simple linear relationship, this model links the expected return of an asset to market risk, making [Markowitz (1952)](https://www.jstor.org/stable/2975974)$^{[11]}$'s portfolio selection theory more relevant to the real world, and laying a theoretical foundation for the establishment of multi-factor stock selection models.

According to the Capital Asset Pricing Model, for a given asset $i$, the relationship between its expected return and the expected return of the market portfolio can be expressed as:

$$
E\left(r_{i}\right) = r_{f} + \beta_{im}\left[E\left(r_{m}\right) - r_{f}\right]
$$

**Where:**

-   $E\left(r_{i}\right)$ is the expected return of asset $i$
-   $r_{f}$ is the risk-free rate
-   $\beta_{im}$ (Beta) is the systematic risk coefficient of asset $i$, $\beta_{im} = \frac{\operatorname{Cov}\left(r_{i}, r_{m}\right)}{\operatorname{Var}\left(r_{m}\right)}$
-   $E\left(r_{m}\right)$ is the expected return of the market portfolio $m$
-   $E\left(r_{m}\right) - r_{f}$ is the market risk premium

#### 2.2.3 Arbitrage Pricing Theory and Multi-Factor Model

[Ross (1976)](https://www.top1000funds.com/wp-content/uploads/2014/05/The-Arbitrage-Theory-of-Capital-Asset-Pricing.pdf)$^{[16]}$ proposed the Arbitrage Pricing Theory (APT). This theory argues that arbitrage behavior is the decisive factor in forming market equilibrium prices. By introducing a series of factors in the return formation process to construct linear correlations, it overcomes the limitations of the Capital Asset Pricing Model (CAPM) and provides important theoretical guidance for subsequent researchers.

Arbitrage Pricing Theory is considered the theoretical basis of the Multi-Factor Model (MFM), an important component of asset pricing models, and one of the cornerstones of asset pricing theory. The general form of the multi-factor model is:

$$
r_{j} = a_{j} + \lambda_{j1} f_{1} + \lambda_{j2} f_{2} + \cdots + \lambda_{jn} f_{n} + \delta_{j}
$$

**Where:**

-   $r_{j}$ is the return of asset $j$
-   $a_{j}$ is a constant for asset $j$
-   $f_{n}$ is the systematic factor
-   $\lambda_{jn}$ is the factor loading
-   $\delta_{j}$ is the random error

#### 2.2.4 Fama-French Three-Factor Model and Five-Factor Model

[Fama (1992) and French (1992)](https://www.bauer.uh.edu/rsusmel/phd/Fama-French_JFE93.pdf)$^{[17]}$ used a combination of cross-sectional regression and time series methods and found that the $\beta$ value of the stock market could not explain the differences in returns of different stocks, while market capitalization, book-to-market ratio, and price-to-earnings ratio of listed companies could significantly explain the differences in stock returns. They argued that excess returns are compensation for risk factors not reflected by $\beta$ in CAPM, and thus proposed the Fama-French Three-Factor Model. The three factors are:

-   **Market Risk Premium Factor** (Market Risk Premium)
    -   Represents the overall systematic risk of the market, i.e., the difference between the expected return of the market portfolio and the risk-free rate.
    -   Measures the excess return investors expect for bearing systematic risk (risk that cannot be eliminated through diversification).
    -   Calculated as:
        $$
        \text{Market Risk Premium} = E(R_m) - R_f
        $$
        where $E(R_m)$ is the expected market return, and $R_f$ is the risk-free rate.

-   **Size Factor** (SMB: Small Minus Big)
    -   Represents the return difference between small-cap stocks and large-cap stocks.
    -   Small-cap stocks are generally riskier, but historical data shows that their expected returns tend to be higher than those of large-cap stocks.
    -   Calculated as:
        $$
        SMB = R_{\text{Small}} - R_{\text{Big}}
        $$
        reflecting the market's compensation for the additional risk premium of small-cap stocks.

-   **Value Factor** (HML: High Minus Low)
    -   Reflects the return difference between high book-to-market ratio stocks (i.e., "value stocks") and low book-to-market ratio stocks (i.e., "growth stocks").
    -   Stocks with high book-to-market ratios are usually priced lower (undervalued by the market), but may achieve higher returns in the long run.
    -   Calculated as:
        $$
        HML = R_{\text{High}} - R_{\text{Low}}
        $$
        Stocks with low book-to-market ratios may be overvalued due to overly optimistic market expectations.

This model concretizes the factors in the APT model and concludes that investing in small-cap, high-growth stocks has the characteristics of high risk and high return. The Fama-French Three-Factor Model is widely used in the analysis and practice of modern investment behavior.

Subsequently, [Fama (2015) and French (2015)](https://tevgeniou.github.io/EquityRiskFactors/bibliography/FiveFactor.pdf)$^{[18]}$ extended the three-factor model, adding the following two factors:

-   **Profitability Factor** (RMW: Robust Minus Weak)
    -   Reflects the return difference between highly profitable companies and less profitable companies.
    -   Companies with strong profitability (high ROE, net profit margin) are more likely to provide stable and higher returns.
    -   Calculated as:
        $$
        RMW = R_{\text{Robust}} - R_{\text{Weak}}
        $$

-   **Investment Factor** (CMA: Conservative Minus Aggressive)
    -   Reflects the return difference between conservative investment companies and aggressive investment companies.
    -   Aggressive companies (rapidly expanding, high capital expenditure) are usually accompanied by greater operational risks, while conservative companies (relatively stable capital expenditure) show higher stability and returns.
    -   Calculated as:
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
-   $R_i$: Expected return of stock $i$
-   $R_f$: Risk-free rate of return
-   $R_m$: Expected return of the market portfolio
-   $R_m - R_f$: Market risk premium factor
-   $SMB$: Return of small-cap stocks minus large-cap stocks
-   $HML$: Return of high book-to-market ratio stocks minus low book-to-market ratio stocks
-   $RMW$: Return of high profitability stocks minus low profitability stocks
-   $CMA$: Return of conservative investment stocks minus aggressive investment stocks
-   $\beta_{i,*}$: Sensitivity of stock $i$ to the corresponding factor
-   $\epsilon_i$: Regression residual

#### 2.2.5 Model Comparison Table

The following table summarizes the core content and factor sources of the **Mean-Variance Model**, **Capital Asset Pricing Model (CAPM)**, **Arbitrage Pricing Theory (APT)**, and **Fama-French Models**:

| **Model**                     | **Core Content**                                                                                             | **Factor Source**                                      |
| :---------------------------- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- |
| **Mean-Variance Model**       | Foundation of portfolio theory, optimizes portfolio through expected returns and covariance matrix.             | Expected returns and covariance matrix of assets in portfolio |
| **Capital Asset Pricing Model (CAPM)** | Explains asset returns through market risk factor ($\beta$), laying the theoretical foundation for multi-factor models. | Market factor $\beta$                                 |
| **Arbitrage Pricing Theory (APT)**    | Multi-factor framework, allows multiple economic variables to explain asset returns, e.g., inflation rate, interest rate. | Multiple factors (macroeconomic variables, e.g., inflation rate, interest rate) |
| **Fama-French Three-Factor Model**  | Adds size factor and book-to-market ratio factor, improving the explanatory power of asset returns.          | Market factor, SMB (size factor), HML (book-to-market ratio factor) |
| **Fama-French Five-Factor Model**   | Adds profitability factor and investment factor on the basis of the three-factor model, further improving asset pricing model. | Market factor, SMB, HML, RMW (profitability factor), CMA (investment factor) |

The following table summarizes the advantages and disadvantages of these models:

| **Model**                     | **Advantages**                                                                                                 | **Disadvantages**                                                                                             |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **Mean-Variance Model**       | Provides a systematic portfolio optimization method, laying the foundation for modern investment theory.          | Only optimizes for return and variance, does not explicitly specify the source of risk compensation.             |
| **Capital Asset Pricing Model (CAPM)** | Simple and easy to use, explains return differences through market risk, provides a theoretical basis for multi-factor models. | Assumes a single factor (market risk) determines returns, ignores other systematic risk factors.                 |
| **Arbitrage Pricing Theory (APT)**    | Allows multiple factors to explain asset returns, reduces reliance on single-factor assumptions, more flexible.          | Does not specify concrete factors, lower operability, only provides a framework.                               |
| **Fama-French Three-Factor Model**  | Significantly improves the explanatory power of asset returns by adding size factor and book-to-market ratio factor.    | Ignores other factors such as profitability and investment behavior.                                          |
| **Fama-French Five-Factor Model**   | More comprehensively captures key variables affecting asset returns by adding profitability factor and investment factor on the basis of the three-factor model. | Higher model complexity, high data requirements, may still miss some potential factors.                               |

## Chapter 3 Comparative Study of LSTM, GRU, and BiLSTM in Stock Price Prediction

### 3.1 Introduction to Experimental Data

Many scholars, both domestically and internationally, focus their research on their own country's stock indices, with relatively less research on individual stocks in different markets. Furthermore, few studies compare LSTM, GRU, and BiLSTM models directly. Therefore, this paper selects Shanghai Pudong Development Bank (SPD Bank, code 600000) in the domestic A-share market and International Business Machines Corporation (IBM) in the US stock market for research. This approach allows for a more accurate comparison of the three models used. For SPD Bank, stock data from January 1, 2008, to December 31, 2020, is used, totaling 3114 valid data points, sourced from the Tushare financial big data platform. We select six features from this dataset for the experiment: date, open price, close price, high price, low price, and volume. For the SPD Bank dataset, all five features except date (used as a time series index) are used as independent variables. For IBM, stock data from January 2, 1990, to November 15, 2018, is used, totaling 7278 valid data points, sourced from Yahoo Finance. We select seven features from this dataset for the experiment: date, open price, high price, low price, close price, adjusted close price (Adj Close), and volume. For the IBM dataset, all six features except date (used as a time series index) are used as independent variables. In this experiment, the closing price (close) is chosen as the variable to be predicted. Tables 3.1.1 and 3.1.2 show partial data from the two datasets.

#### 3.1.1 Partial Display of SPD Bank Dataset

| date       | open  | close | high  | low   | volume      | code   |
|------------|-------|-------|-------|-------|-------------|--------|
| 2008-01-02 | 9.007 | 9.101 | 9.356 | 8.805 | 131583.90   | 600000 |
| 2008-01-03 | 9.007 | 8.645 | 9.101 | 8.426 | 211346.56   | 600000 |
| 2008-01-04 | 8659  | 9.009 | 9.111 | 8.501 | 139249.67   | 600000 |
| 2008-01-07 | 8.970 | 9.515 | 9.593 | 8.953 | 228043.01   | 600000 |
| 2008-01-08 | 9.551 | 9.583 | 9.719 | 9.517 | 161255.31   | 600000 |
| 2008-01-09 | 9.583 | 9.663 | 9.772 | 9.432 | 102510.92   | 600000 |
| 2008-01-10 | 9.701 | 9.680 | 9.836 | 9.602 | 217966.25   | 600000 |
| 2008-01-11 | 9.670 | 10.467| 10.532| 9.670 | 231544.21   | 600000 |
| 2008-01-14 | 10.367| 10.059| 10.433| 10.027| 142918.39   | 600000 |
| 2008-01-15 | 10.142| 10.051| 10.389| 10.006| 161221.52   | 600000 |

**Data Source**: [Tushare](https://github.com/waditu/tushare)

#### 3.1.2 Partial Display of IBM Dataset

| Date       | Open    | High    | Low     | Close   | Adj Close | Volume  |
|------------|---------|---------|---------|---------|-----------|---------|
| 1990-01-02 | 23.6875 | 24.5313 | 23.6250 | 24.5000 | 6.590755  | 7041600 |
| 1990-01-03 | 24.6875 | 24.8750 | 24.5938 | 24.7188 | 6.649599  | 9464000 |
| 1990-01-04 | 24.7500 | 25.0938 | 24.7188 | 25.0000 | 6.725261  | 9674800 |
| 1990-01-05 | 24.9688 | 25.4063 | 24.8750 | 24.9375 | 6.708448  | 7570000 |
| 1990-01-08 | 24.8125 | 25.2188 | 24.8125 | 25.0938 | 6.750481  | 4625200 |
| 1990-01-09 | 25.1250 | 25.3125 | 24.8438 | 24.8438 | 6.683229  | 7048000 |
| 1990-01-10 | 24.8750 | 25.0000 | 24.6563 | 24.7500 | 6.658009  | 5945600 |
| 1990-01-11 | 24.8750 | 25.0938 | 24.8438 | 24.9688 | 6.716855  | 5905600 |
| 1990-01-12 | 24.6563 | 24.8125 | 24.4063 | 24.4688 | 6.582347  | 5390800 |
| 1990-01-15 | 24.4063 | 24.5938 | 24.3125 | 24.5313 | 6.599163  | 4035600 |

**Data Source**: [Yahoo Finance](https://finance.yahoo.com/quote/IBM/history/)

### 3.2 Experimental Data Preprocessing

#### 3.2.1 Data Normalization

In the experiment, there are differences in units and magnitudes among various features. For example, the magnitude difference between stock prices and trading volume is huge, which will affect the final prediction results of our experiment. Therefore, we use the `MinMaxScaler` method from the `sklearn.preprocessing` library to scale the features of the data to between 0 and 1. This can not only improve the model accuracy but also increase the model convergence speed. The normalization formula is:

$$
x^{\prime}=\frac{x-\min (x)}{\max (x)-\min (x)}
$$

where $x^{\prime}$ is the normalized data, $x$ is the original data, $\min (x)$ is the minimum value of the original dataset, and $\max (x)$ is the maximum value of the original dataset. After obtaining the prediction results in our experimental process, we also need to denormalize the data before we can perform stock price prediction and model evaluation.

#### 3.2.2 Data Partitioning

Here, the entire experimental datasets of SPD Bank and IBM are input respectively, and the timestep of the recurrent kernel is set to 60 for both, with the number of input features per timestep being 5 and 6, respectively. This allows inputting data from the previous 60 trading days to predict the closing price on the 61st day. This makes our dataset meet the input requirements of the three neural network models to be compared later, which are the number of samples, the number of recurrent kernel unfolding steps, and the number of input features per timestep. Then, we divide the normalized SPD Bank dataset into training, validation, and test sets in a ratio of 2488:311:255. The normalized IBM dataset is divided into training, validation, and test sets in a ratio of 6550:364:304. We partition out a validation set here to facilitate adjusting the hyperparameters of the models to optimize each model before comparison.

### 3.3 Model Network Structure

The network structures of each model set in this paper through a large number of repeated experiments are shown in the table below. The default tanh and linear activation functions of recurrent neural networks are used between layers, and Dropout is added to prevent overfitting. The dropout rate is set to 0.2. The number of neurons in each recurrent layer of LSTM and GRU is 50, and the number of neurons in the recurrent layer of BiLSTM is 100. Each model of LSTM, GRU, and BiLSTM adopts four layers of LSTM, GRU, BiLSTM, and one fully connected layer, with a Dropout set between each network layer.

#### 3.3.1 LSTM Network Structure for IBM

| Layer(type)         | Output Shape | Param# |
|---------------------|--------------|--------|
| lstm_1 (LSTM)       | (None, 60, 50)| 11400  |
| dropout_1 (Dropout) | (None, 60, 50)| 0      |
| lstm_2 (LSTM)       | (None, 60, 50)| 20200  |
| dropout_2 (Dropout) | (None, 60, 50)| 0      |
| lstm_3 (LSTM)       | (None, 60, 50)| 20200  |
| dropout_3 (Dropout) | (None, 60, 50)| 0      |
| lstm_4 (LSTM)       | (None, 50)    | 20200  |
| dropout_4 (Dropout) | (None, 50)    | 0      |
| dense_1 (Dense)     | (None, 1)     | 51     |

**Total params**: 72,051  
**Trainable params**: 72,051  
**Non-trainable params**: 0

---

#### 3.3.2 GRU Network Structure for IBM

| Layer(type)         | Output Shape | Param# |
|---------------------|--------------|--------|
| gru_1 (GRU)         | (None, 60, 50)| 8550   |
| dropout_1 (Dropout) | (None, 60, 50)| 0      |
| gru_2 (GRU)         | (None, 60, 50)| 15150  |
| dropout_2 (Dropout) | (None, 60, 50)| 0      |
| gru_3 (GRU)         | (None, 60, 50)| 15150  |
| dropout_3 (Dropout) | (None, 60, 50)| 0      |
| gru_4 (GRU)         | (None, 50)    | 15150  |
| dropout_4 (Dropout) | (None, 50)    | 0      |
| dense_1 (Dense)     | (None, 1)     | 51     |

**Total params**: 54,051  
**Trainable params**: 54,051  
**Non-trainable params**: 0

---

#### 3.3.3 BiLSTM Network Structure for IBM

| Layer(type)                   | Output Shape | Param# |
|-------------------------------|--------------|--------|
| bidirectional_1 (Bidirection) | (None, 60, 100)| 22800  |
| dropout_1 (Dropout)           | (None, 60, 100)| 0      |
| bidirectional_2 (Bidirection) | (None, 60, 100)| 60400  |
| dropout_2 (Dropout)           | (None, 60, 100)| 0      |
| bidirectional_3 (Bidirection) | (None, 60, 100)| 60400  |
| dropout_3 (Dropout)           | (None, 60, 100)| 0      |
| bidirectional_4 (Bidirection) | (None, 100)   | 60400  |
| dropout_4 (Dropout)           | (None, 100)   | 0      |
| dense_1 (Dense)               | (None, 1)     | 101    |

**Total params**: 204,101  
**Trainable params**: 204,101  
**Non-trainable params**: 0

### 3.4 Model Compilation and Hyperparameter Settings

In this paper, after continuous hyperparameter tuning with the goal of minimizing the loss function on the validation set, the following hyperparameters are selected for the three models of SPD Bank: `epochs=100`, `batch_size=32`; and for the three models of IBM: `epochs=50`, `batch_size=32`. The optimizer used is Adaptive Moment Estimation [(Adam)](https://arxiv.org/abs/1412.6980)$^{[19]}$. The default values in its `keras` package are used, i.e., `lr=0.001`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-08`, and `decay=0.0`. The loss function is Mean Squared Error (MSE).

**Parameter Explanation:**

-   `lr`: Learning rate
-   `beta_1`: Exponential decay rate for the first moment estimate
-   `beta_2`: Exponential decay rate for the second moment estimate
-   `epsilon`: Fuzz factor
-   `decay`: Learning rate decay value after each update

### 3.5 Experimental Results and Analysis

First, let's briefly introduce the evaluation metrics used for the models. The calculation formulas are as follows:

1.  **Mean Squared Error (MSE)**:

$$
M S E=\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}
$$

2.  **Root Mean Squared Error (RMSE)**:

$$
R M S E=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}}
$$

3.  **Mean Absolute Error (MAE)**:

$$
M A E=\frac{1}{n} \sum_{i=1}^{n}\left|Y_{i}-\hat{Y}_{i}\right|
$$

4.  **\( R^2 \) (R Squared)**:

$$
\begin{gathered}
\bar{Y}=\frac{1}{n} \sum_{i=1}^{n} Y_{i} \\
R^{2}=1-\frac{\sum_{i=1}^{n}\left(Y_{i}-\hat{Y}_{i}\right)^{2}}{\sum_{i=1}^{n}\left(Y_{i}-\bar{Y}\right)^{2}}
\end{gathered}
$$

Where: $n$ is the number of samples, $Y_{i}$ is the actual closing price of the stock, $\hat{Y}_{i}$ is the predicted closing price of the stock, and $\bar{Y}$ is the average closing price of the stock. The smaller the MSE, RMSE, and MAE, the more accurate the model. The larger the \( R^2 \), the better the goodness of fit of the model coefficients.

#### 3.5.1 Experimental Results for SPD Bank

|               | LSTM     | GRU      | BiLSTM   |
|---------------|----------|----------|----------|
| **MSE**       | 0.059781 | 0.069323 | 0.056454 |
| **RMSE**      | 0.244501 | 0.263292 | 0.237601 |
| **MAE**       | 0.186541 | 0.202665 | 0.154289 |
| **R-squared** | 0.91788  | 0.896214 | 0.929643 |

Comparing the evaluation metrics of the three models, we can find that on the SPD Bank test set, the MSE, RMSE, and MAE of the BiLSTM model are smaller than those of the LSTM and GRU models, while the R-Squared is larger than those of the LSTM and GRU models. By comparing RMSE, we find that BiLSTM has a 2.90% performance improvement over LSTM and a 10.81% performance improvement over GRU on the validation set.

#### 3.5.2 Experimental Results for IBM

|               | LSTM     | GRU      | BiLSTM   |
|---------------|----------|----------|----------|
| **MSE**       | 18.01311 | 12.938584| 11.057501|
| **RMSE**      | 4.244186 | 3.597024 | 3.325282 |
| **MAE**       | 3.793223 | 3.069033 | 2.732075 |
| **R-squared** | 0.789453 | 0.851939 | 0.883334 |

Comparing the evaluation metrics of the three models, we can find that on the IBM test set, the MSE, RMSE, and MAE of the BiLSTM model are smaller than those of the LSTM and GRU models, while the R-Squared is larger than those of the LSTM and GRU models. By comparing RMSE, we find that BiLSTM has a 27.63% performance improvement over LSTM and an 8.17% performance improvement over GRU on the validation set.

### 3.6 Chapter Summary

This chapter first introduced the SPD Bank and IBM datasets and the features used in the experiment. Then, it performed preprocessing steps of data normalization and data partitioning on the datasets. It also detailed the network structures and hyperparameters of the LSTM, GRU, and BiLSTM models used in the experiment. Finally, it obtained the loss function images and a series of fitting graphs for each model. By comparing multiple evaluation metrics and fitting images of the models, it is concluded that the BiLSTM model can better predict stock prices, laying a foundation for our next chapter's research on the LightGBM-BiLSTM quantitative investment strategy.

---

## Chapter 4 Research on Quantitative Investment Model Based on LightGBM-BiLSTM

### 4.1 Experimental Steps

{{< figure
    src="LightGBM_BiLSTM_Flow.png"
    caption="Fig. 11. LightGBM-BiLSTM Diagram."
    align="center"
>}}

As shown in Fig. 11, this experiment first selects 50 factors from the factor library. Then, it performs factor cleaning steps of outlier removal, standardization, and missing value imputation on the factors sequentially. Next, the LightGBM model is used for factor selection, and the top ten factors with the highest importance are selected as the factors for this cross-sectional selection. Subsequently, a BiLSTM model is used to establish a multi-factor model, and finally, backtesting analysis is performed.

### 4.2 Experimental Data

The market data used in this paper comes from [Tushare](https://github.com/waditu/tushare). The specific features of the dataset are shown in the table below.

#### 4.2.1 Features Included in the Stock Dataset

| Name           | Type  | Description                                     |
| :--------------- | :---- | :---------------------------------------------- |
| ts_code        | str   | Stock code                                      |
| trade_date     | str   | Trading date                                    |
| open           | float | Open price                                      |
| high           | float | High price                                      |
| low            | float | Low price                                       |
| close          | float | Close price                                     |
| pre_close      | float | Previous close price                            |
| change         | float | Change amount                                   |
| pct_chg        | float | Change percentage (unadjusted)                  |
| vol            | float | Volume (in hands)                               |
| amount         | float | Turnover (in thousands of CNY)                  |

The A-share market-wide daily dataset contains 5,872,309 rows of data, i.e., 5,872,309 samples. As shown in Table 4.2.1, the A-share market-wide daily dataset has the following 11 features, in order: stock code (ts_code), trading date (trade_date), open price (open), high price (high), low price (low), close price (close), previous close price (pre_close), change amount (change), turnover rate (turnover_rate), turnover amount (amount), total market value (total_mv), and adjustment factor (adj_factor).

#### 4.2.2 Partial Display of A-Share Market-Wide Daily Dataset

| ts_code     | trade_date | open  | high  | low   | close | pre_close | change | vol      | amount      |
| :------------ | :----------- | :---- | :---- | :---- | :---- | :---------- | :----- | :--------- | :------------ |
| 600613.SH   | 20120104   | 8.20  | 8.20  | 7.84  | 7.86  | 8.16      | -0.30  | 4762.98  | 3854.1000   |
| 600690.SH   | 20120104   | 9.00  | 9.17  | 8.78  | 8.78  | 8.93      | -0.15  | 142288.41| 127992.6050 |
| 300277.SZ   | 20120104   | 22.90 | 22.98 | 20.81 | 20.88 | 22.68     | -1.80  | 12212.39 | 26797.1370  |
| 002403.SZ   | 20120104   | 8.87  | 8.90  | 8.40  | 8.40  | 8.84      | -0.441 | 10331.97 | 9013.4317   |
| 300179.SZ   | 20120104   | 19.99 | 20.32 | 19.20 | 19.50 | 19.96     | -0.46  | 1532.31  | 3008.0594   |
| 600000.SH   | 20120104   | 8.54  | 8.56  | 8.39  | 8.41  | 8.49      | -0.08  | 342013.79| 290229.5510 |
| 300282.SZ   | 20120104   | 22.90 | 23.33 | 21.02 | 21.02 | 23.35     | -2.33  | 38408.60 | 86216.2356  |
| 002319.SZ   | 20120104   | 9.74  | 9.95  | 9.38  | 9.41  | 9.73      | -0.32  | 4809.74  | 4671.4803   |
| 601991.SH   | 20120104   | 5.17  | 5.39  | 5.12  | 5.25  | 5.16      | 0.09   | 145268.38| 76547.7490  |
| 000780.SZ   | 20120104   | 10.42 | 10.49 | 10.00 | 10.00 | 10.30     | -0.30  | 20362.30 | 20830.1761  |

**[5872309 rows x 11 columns]**

The CSI All Share daily dataset contains 5,057 rows of data, i.e., 5,057 samples. As shown in Table 4.2.2, the CSI All Share daily dataset has the following 7 features, in order: trading date (trade_date), open price (open), high price (high), low price (low), close price (close), volume (volume), and previous close price (pre_close).

#### 4.2.3 Partial Display of CSI All Share Daily Dataset

| trade_date | open      | high      | low       | close     | volume         | pre_close |
| :----------- | :-------- | :-------- | :-------- | :-------- | :------------- | :-------- |
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

**[5057 rows x 7 columns]**

Table 4.2.4 below shows partial data of the original factors. After sequentially going through the four factor cleaning steps of missing value imputation, outlier removal, factor standardization, and factor neutralization mentioned above, partial data of the cleaned factors are obtained as shown in Table 4.2.5.

#### 4.2.4 Original Factor Data

| trade_date | sec_code    | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
| :----------- | :------------ | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :-- |
| 2005-01-04 | 600874.SH   | 0.001684 | NaN      | 9.445412 | 9.445412 | 9.445408 | -1.00    | NaN      | 12651.124023 | ... |
| 2005-01-04 | 000411.SZ   | 0.021073 | NaN      | 5.971262 | 5.971262 | 5.971313 | 0.38     | NaN      | 392.124298 | ... |
| 2005-01-04 | 000979.SZ   | 0.021207 | NaN      | 6.768918 | 6.768918 | 6.768815 | -1.45    | NaN      | 870.587585 | ... |
| 2005-01-04 | 000498.SZ   | 0.030220 | NaN      | 8.852752 | 8.852752 | 8.852755 | 0.55     | NaN      | 6994.011719 | ... |
| 2005-01-04 | 600631.SH   | 0.015699 | NaN      | 9.589897 | 9.589897 | 9.589889 | -1.70    | NaN      | 14616.806641 | ... |

#### 4.2.5 Cleaned Factor Data

| sec_code  | trade_date | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
| :---------- | :----------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :-- |
| 000001.SZ | 2005-01-04 | -1.58653 | 0.01545  | 1.38306  | 1.38306  | 1.38306  | 0.13392  | 0.01545  | 1.38564  | ... |
| 000002.SZ | 2005-01-04 | 1.36761  | -0.44814 | 1.69728  | 1.69728  | 1.69728  | 1.04567  | -0.44814 | 1.69728  | ... |
| 000004.SZ | 2005-01-04 | 0.32966  | -1.41654 | -0.13907 | -0.13907 | -0.13907 | -0.34769 | -1.41654 | -0.13650  | ... |
| 000005.SZ | 2005-01-04 | 0.61297  | -1.13066 | 1.05339  | 1.05339  | 1.05339  | -1.20020 | -1.13066 | 1.05597  | ... |
| 000006.SZ | 2005-01-04 | -0.35542 | 1.67667  | -0.07726 | -0.07726 | -0.07726 | 1.55820  | 1.67667  | -0.07469  | ... |

#### 4.2.6 Factor Data

-   **Construction of Price-Volume Factors**

This paper uses the following method to construct price-volume factors. There are two basic elements for constructing price-volume factors: first, **basic fields**, and second, **operators**. As shown in Table 4.2.1, basic fields include daily frequency high price, low price, open price, close price, previous day's close price, volume, change percentage, turnover rate, turnover amount, total market value, and adjustment factor.

#### 4.2.7 Basic Field Table

| No. | Field Name    | Meaning                                        |
| :---: | :------------ | :--------------------------------------------- |
| high  | High Price    | Highest price in intraday transactions         |
| low   | Low Price     | Lowest price in intraday transactions          |
| open  | Open Price    | Price at which the call auction concludes       |
| close | Close Price   | Price of the last transaction of the day       |
| pre_close | Previous Close Price | Price of the last transaction of the previous day |
| vol   | Volume        | Number of shares traded throughout the day     |
| pct_chg | Change Percentage | Percentage change of the security for the day   |
| turnover_rate | Turnover Rate | Turnover rate of the security for the day    |
| amount | Turnover Amount | Total value of transactions for the day       |
| total_mv | Total Market Value | Total value of the stock, calculated by total shares outstanding multiplied by the current stock price |
| adj_factor | Adjustment Factor | Ratio for adjusting for dividends and splits   |

This paper obtains the operator list shown in the table below through the basic operator set provided by [gplearn](https://github.com/trevorstephens/gplearn) and some self-defined special operators.

#### 4.2.8 Operator List

| Operator                    | Name          | Definition                                                     |
| :-------------------------- | :----------- | :------------------------------------------------------------ |
| add(x, y)                  | Sum           | \( x + y\); element-wise operation                             |
| \(\operatorname{div}(x, y)\) | Division      | \( x / y\); element-wise operation                             |
| \(\operatorname{mul}(x, y)\) | Multiplication | \( x \cdot y\); element-wise operation                         |
| \(\operatorname{sub}(x, y)\) | Subtraction   | \( x - y\); element-wise operation                             |
| neg(x)                     | Negative      | \(-x\); element-wise operation                                 |
| \(\log(x)\)                | Logarithm     | \(\log(x)\); element-wise operation                           |
| max(x, y)                  | Maximum       | Larger value between \(x\) and \(y\); element-wise operation |
| \(\min(x, y)\)              | Minimum       | Smaller value between \(x\) and \(y\); element-wise operation |
| delta_d(x)                 | d-day Difference | Current day's \(x\) value minus \(x\) value \(d\) days ago; **time series operation** |
| delay_d(x)                 | d-day Delay   | \(x\) value \(d\) days ago; **time series operation**         |
| Corr_d(x, y)               | d-day Correlation | Correlation between \(x\) values and \(y\) values over \(d\) days; **time series operation** |
| Max_d(x)                   | d-day Maximum   | Maximum value of \(x\) over \(d\) days; **time series operation** |
| Min_d(x)                   | d-day Minimum   | Minimum value of \(x\) over \(d\) days; **time series operation** |
| sort_d(x)                  | d-day Rank      | Rank of \(x\) values over \(d\) days; **time series operation** |
| Argsortmin_d(x)            | d-day Minimum Position | Position of the minimum value of \(x\) over \(d\) days; **time series operation** |
| Argsortmax_d(x)            | d-day Maximum Position | Position of the maximum value of \(x\) over \(d\) days; **time series operation** |
| \(\operatorname{inv}(x)\)         | Inverse       | \( 1 / x\); element-wise operation                             |
| Std_d(x)                   | d-day Standard Deviation | Standard deviation of \(x\) values over \(d\) days; **time series operation** |
| abs(x)                     | Absolute Value | \(\lvert x\rvert\); element-wise operation                     |

#### 4.2.9 Genetic Programming

The core idea of **Genetic Programming (GP)** is to use evolutionary algorithms to automatically "evolve" factor expressions with strong predictive power in the vast search space composed of operators and basic fields. For factor mining in this paper, the main goal of GP is to **search** and find those factors that can better predict future stock returns from all possible expressions that can be combined from the **basic fields** in Table 4.2.7 and the **operators** in Table 4.2.8. The core process of GP can be divided into the following steps:

##### Initialization

1.  **Define Operator Set and Basic Fields**
    -   Operator set (operators) as shown in Table 4.2.8, including operations such as addition, subtraction, multiplication, division, logarithm, absolute value, delay, moving maximum/minimum, moving correlation coefficient, etc.
    -   Basic fields (terminals) as shown in Table 4.2.7, including open price, close price, high price, low price, volume, adjustment factor, etc.
    These operators and basic fields can be regarded as "nodes" in the factor expression tree, where basic fields are leaf nodes (terminal nodes), and operators are internal nodes.

2.  **Randomly Generate Initial Population**
    -   In the initialization phase, based on the given operator set and field set, a series of factor expressions (which can be represented as several syntax trees or expression trees) are randomly "spliced" to form an initial population.
    -   For example, it may randomly generate
        \[
          \text{Factor 1}: \mathrm{Max\_5}\bigl(\mathrm{add}(\mathrm{vol}, \mathrm{close})\bigr), \quad
          \text{Factor 2}: \mathrm{sub}\bigl(\mathrm{adj\_factor}, \mathrm{neg}(\mathrm{turnover\_rate})\bigr),
          \dots
        \]
    -   Each factor expression will correspond to an individual.

##### Fitness Function

1.  **Measure Factor's Predictive Ability**
    -   For each expression (individual), we need to evaluate its predictive ability for future returns or other objectives. Specifically, we can calculate the **correlation coefficient** (IC) or a more comprehensive indicator IR (Information Ratio) between the **next period's stock return** \( r^{T+1} \) and the current period's factor exposure \( x_k^T \) to measure it.

2.  **Set Objective**
    -   If we want the factor to have a higher correlation (IC), we can set the fitness function to \(\lvert \rho(x_k^T, r^{T+1})\rvert\);
    -   If we want the factor to have a higher IR, we can set the fitness function to the IR value.
    -   The higher the factor IC or IR, the higher the "fitness" of the expression.

Therefore, we usually set:
\[
\text{Fitness} \bigl(F(x)\bigr) \;=\;
\begin{cases}
\lvert \rho(x_k^T, r^{T+1})\rvert \quad &\text{(Maximize IC)},\\[6pt]
\mathrm{IR}(x_k^T) \quad &\text{(Maximize IR)}.
\end{cases}
\]
where \(\rho(\cdot)\) represents the correlation coefficient, and \(\mathrm{IR}(\cdot)\) is the IR indicator.

##### Selection, Crossover, and Mutation

1.  **Selection**
    -   Based on the results of the fitness function, expressions with high factor fitness are "retained" or "bred", while expressions with lower fitness are eliminated.
    -   This is similar to "survival of the fittest" in biological evolution.

2.  **Crossover**
    -   Randomly select a part of the "nodes" of several expressions with higher fitness (parents) for exchange, so as to obtain new expressions (offspring).
    -   In the expression tree structure, subtree A and subtree B can be interchanged to generate new offspring expressions.
    -   For example, if a subtree of expression tree \(\mathrm{FactorA}\) is exchanged with the corresponding subtree of expression tree \(\mathrm{FactorB}\), two new expressions are generated.

3.  **Mutation**
    -   Randomly change some nodes of the expression with a certain probability, such as:
        -   Replacing the operator of the node (for example, changing \(\mathrm{add}\) to \(\mathrm{sub}\)),
        -   Replacing the basic field of the terminal node (for example, changing \(\mathrm{vol}\) to \(\mathrm{close}\)),
        -   Or randomly changing operation parameters (such as moving window length, smoothing factor, etc.).
    -   Mutation can increase the diversity of the population and avoid premature convergence or falling into local optima.

##### Iterative Evolution

1.  **Iterative Execution**
    -   Repeatedly execute selection, crossover, and mutation operations for multiple generations.
    -   Each generation produces a new population of factor expressions and evaluates their fitness.

2.  **Convergence and Termination**
    -   When evolution reaches a predetermined stopping condition (such as the number of iterations, fitness threshold, etc.), the algorithm terminates.
    -   Usually, we will select **several** factor expressions with higher final fitness and regard them as the evolution results.

##### Mathematical Representation: Searching for Optimal Factor Expressions

Abstracting the above process into the following formula, the factor search objective can be simply expressed as:

\[
F(x) \;=\; \mathrm{GP}\bigl(\{\text{operators}\}, \{\text{terminals}\}\bigr),
\]
indicating that a function \(F(x)\) is searched through the GP algorithm on a given operator set (operators) and basic field set (terminals). From the perspective of optimization, we hope to find:

\[
\max_{F} \bigl\lvert \rho(F^T, r^{T+1}) \bigr\rvert
\quad \text{or} \quad
\max_{F} \; \mathrm{IR}\bigl(F\bigr),
\]
where
-   \(\rho(\cdot)\) represents the correlation coefficient (IC) between the factor and the next period's return,
-   \(\mathrm{IR}(\cdot)\) represents the IR indicator of the factor.

In practical applications, we will give a backtesting period, score the candidate factors of each generation (IC/IR evaluation), and continuously "evolve" better factors through the iterative process of selection, crossover, and mutation.

**Through the above steps, we can finally automatically mine a batch of factor expressions that have strong predictive power for future returns and good robustness (such as higher IR) in the vast search space of operator combinations and basic field combinations.**

#### 4.2.10 Partially Mined Factors

| Factor Name | Definition                                     |
| :---------- | :--------------------------------------------- |
| 0         | Max＿25(add(turnover_rate, vol))              |
| 1         | Max＿30(vol)                                   |
| 2         | Max＿25(turnover_rate)                         |
| 3         | Max＿35(add(vol, close))                       |
| 4         | Max＿30(turnover_rate)                         |
| 5         | sub(Min＿20(neg(pre_close)), div(vol, adj_factor)) |
| 6         | Max＿60(max(vol, adj_factor))                  |
| 7         | Max＿50(amount)                                |
| 8         | div(vol, neg(close))                            |
| 9         | min(ArgSortMin＿25(pre_close), neg(vol))        |
| 10        | neg(max(vol, turnover_rate))                  |
| 11        | mul(amount, neg(turnover_rate))                |
| 12        | inv(add(ArgSortMax＿40(change), inv(pct_chg)))  |
| 13        | Std＿40(inv(abs(sub(mul(total_mv, change), min(adj_factor, high)))) |
| 14        | div(log(total_mv),amount)                      |
| 15        | div(neg(Max＿5(amount)), Min＿20(ArgSort＿60(high))) |
| 16        | Corr＿30(inv(abs(sub(mul(total_mv, change), min(adj_factor, high)))), add(log(Max＿10(pre_close)), high)) |
| 17        | ArgSort＿60(neg(turnover_rate))                |
| ...       | ...                                            |

These factors are all obtained by combining from the operator list (Table 4.2.8) and the basic field list (Table 4.2.7) through genetic programming and have different mathematical expressions.

-   **Factor Validity Test**

After we get the mined factors, we need to test the validity of the factors. Common test indicators are **Information Coefficient (IC)** and **Information Ratio (IR)**.
-   **Information Coefficient (IC)** describes the **linear correlation** between the **next period's return rate** of the selected stocks and the current period's factor exposure, which can reflect the robustness of the factor in predicting returns.
-   **Information Ratio (IR)** is the ratio of the mean of excess returns to the standard deviation of excess returns. The information ratio is similar to the Sharpe ratio. The main difference is that the Sharpe ratio uses the risk-free return as a benchmark, while the information ratio uses a risk index as a benchmark. The Sharpe ratio helps to determine the absolute return of a portfolio, and the information ratio helps to determine the relative return of a portfolio. After we calculate the IC, we can calculate the IR based on the IC value. When the IR is greater than 0.5, the factor has a strong ability to stably obtain excess returns.

In actual calculation, the \( \mathrm{IC} \) value of factor \(k\) generally refers to the correlation coefficient between the exposure \( x_k^T \) of factor \(k\) in period \(T\) of the selected stocks and the return rate \( r^{T+1} \) of the selected stocks in period \(T+1\); the \( \mathrm{IR} \) value of factor \(k\) is the mean of the \( \mathrm{IC} \) of factor \(k\) divided by the standard deviation of the \( \mathrm{IC} \) of factor \(k\). The calculation formulas are as follows:

$$
\begin{gathered}
I C=\rho_{x_{k}^{T}, r^{T+1}}=\frac{\operatorname{cov}\left(x_{k}^{T}, r^{T+1}\right)}{\sigma_{x_{k}^{T}} \sigma_{r^{T+1}}}=\frac{\mathrm{E}\left(x_{k}^{T} * r^{T+1}\right)-\mathrm{E}\left(x_{k}^{T}\right) \mathrm{E}\left(r^{T+1}\right)}{\sqrt{\mathrm{E}\left(\left(x_{k}^{T}\right)^{2}\right)-\mathrm{E}\left(x_{k}^{T}\right)^{2}} \cdot \sqrt{\mathrm{E}\left(\left(r^{T+1}\right)^{2}\right)-\mathrm{E}\left(r^{T+1}\right)^{2}}} \\
I R=\frac{\overline{I C}}{\sigma_{I C}}
\end{gathered}
$$

Where:
* $x_{k}^{T}$: the exposure of the selected stock to factor $k$ in period $T$
* $r^{T+1}$: the return of the selected stock in period $T+1$
* $\overline{IC}$: the mean of the Information Coefficient (IC)

This paper uses IR to judge the quality of factors. Through "screening" a large number of different combinations of operators and basic data and IC and IR, this paper obtains the 50 price-volume factors selected in this paper. After IR testing, the table shown in the figure below is obtained by sorting IR from high to low. From the table below, we can see that the IRs of the selected 50 price-volume factors are all greater than 0.5, indicating that these factors have a strong ability to stably obtain excess returns.

#### 4.2.11 Factor IR Test Table

| Factor Name | IR   | Factor Name | IR   |
| :---------- | :--- | :---------- | :--- |
| 0         | 3.11 | 25        | 2.73 |
| 1         | 2.95 | 26        | 2.71 |
| 2         | 2.95 | 27        | 2.70 |
| 3         | 2.95 | 28        | 2.69 |
| 4         | 2.95 | 29        | 2.69 |
| 5         | 2.94 | 30        | 2.69 |
| 6         | 2.94 | 31        | 2.68 |
| 7         | 2.94 | 32        | 2.68 |
| 8         | 2.93 | 33        | 2.68 |
| 9         | 2.93 | 34        | 2.68 |
| 10        | 2.93 | 35        | 2.67 |
| 11        | 2.92 | 36        | 2.67 |
| 12        | 2.91 | 37        | 2.66 |
| 13        | 2.89 | 38        | 2.65 |
| 14        | 2.86 | 39        | 2.65 |
| 15        | 2.83 | 40        | 2.65 |
| 16        | 2.83 | 41        | 2.65 |
| 17        | 2.83 | 42        | 2.64 |
| 18        | 2.79 | 43        | 2.63 |
| 19        | 2.78 | 44        | 2.63 |
| 20        | 2.78 | 45        | 2.62 |
| 21        | 2.76 | 46        | 2.62 |
| 22        | 2.75 | 47        | 2.62 |

It can be seen from this table that among the screened factors, the IRs of all factors are greater than 0.5, which has a strong and stable ability to obtain excess returns.

### 4.3 Factor Cleaning

#### 4.3.1 Factor Missing Value Handling and Outlier Removal

Methods for handling missing values of factors include case deletion, mean imputation, regression imputation, and other methods. This paper adopts a relatively simple mean imputation method to handle missing values, that is, using the average value of the factor to replace the missing data. Methods for factor outlier removal include median outlier removal, percentile outlier removal, and $3 \sigma$ outlier removal. This paper uses the $3 \sigma$ outlier removal method. This method uses the $3 \sigma$ principle in statistics to convert outlier factors that are more than three standard deviations away from the mean of the factor to a position that is just three standard deviations away from the mean. The specific calculation formula is as follows:

$$
X_i^{\prime}= \begin{cases} \bar{X}+3 \sigma & \text{if } X_i > \bar{X} + 3 \sigma \\ \bar{X}-3 \sigma & \text{if } X_i < \bar{X} - 3 \sigma \\ X_i & \text{if } \bar{X} - 3 \sigma < X_i < \bar{X} + 3 \sigma \end{cases}
$$

Where:

-   $X_{i}$: Value of the factor before processing
-   $\bar{X}$: Mean of the factor sequence
-   $\sigma$: Standard deviation of the factor sequence
-   $X_{i}^{\prime}$: Value of the factor after outlier removal

#### 4.3.2 Factor Standardization

In this experiment, multiple factors are selected, and the dimensions of each factor are not completely the same. For the convenience of comparison and regression, we also need to standardize the factors. Currently, common specific standardization methods include Min-Max standardization, Z-score standardization, and Decimal scaling standardization. This paper chooses the Z-score standardization method. The data is standardized through the mean and standard deviation of the original data. The processed data conforms to the standard normal distribution, that is, the mean is 0 and the standard deviation is 1. The standardized numerical value is positive or negative, and a standard normal distribution curve is obtained.

The Z-score standardization formula used in this paper is as follows:

$$
\tilde{x}=\frac{x_{i}-u}{\sigma}
$$

Where:
-   $x_{i}$: Original value of the factor
-   $u$: Mean of the factor sequence
-   $\sigma$: Standard deviation of the factor sequence
-   $\tilde{x}$: Standardized factor value

#### 4.3.3 Factor Neutralization

Factor neutralization is to eliminate the influence of other factors on our selected factors, so that the stocks selected by our quantitative investment strategy portfolio are more dispersed, rather than concentrated in specific industries or market capitalization stocks. It can better share the risk of the investment portfolio and solve the problem of factor multicollinearity. Market capitalization and industry are the two main independent variables that affect stock returns. Therefore, in the process of factor cleaning, the influence of market capitalization and industry must also be considered. In this empirical study, we adopt the method of only including industry factors and including market factors in industry factors. The single-factor regression model for factors is shown in formula (31). We take the residual term of the following regression model as the new factor value after factor neutralization.

$$
\tilde{r}_{j}^{t}=\sum_{s=1}^{s} X_{j s}^{t} \tilde{f}_{s}^{t}+X_{j k}^{t} \tilde{f}_{k}^{t}+\tilde{u}_{j}^{t}
$$

Where:

-   $\tilde{r}_{j}^{t}$: Return rate of stock $j$ in period $t$
-   $X_{j s}^{t}$: Exposure of stock $j$ in industry $s$ in period $t$
-   $\tilde{f}_{s}^{t}$: Return rate of the industry in period $t$
-   $X_{j k}^{t}$: Exposure of stock $j$ on factor $k$ in period $t$
-   $\tilde{f}_{k}^{t}$: Return rate of factor $k$ in period $t$
-   $\tilde{u}_j^t$: A $0-1$ dummy variable, that is, if stock $j$ belongs to industry $s$, the exposure is 1, otherwise it is 0

In this paper, the industry to which a company belongs is not proportionally split, that is, stock $j$ can only belong to a specific industry $s$, the exposure in industry $s$ is 1, and the exposure in all other industries is 0. This paper uses the Shenwan Hongyuan industry classification standard. The specific classifications are sequentially: agriculture, forestry, animal husbandry and fishery, mining, chemical industry, steel, nonferrous metals, electronic components, household appliances, food and beverage, textile and apparel, light industry manufacturing, pharmaceutical and biological, public utilities, transportation, real estate, commercial trade, catering and tourism, comprehensive, building materials, building decoration, electrical equipment, national defense and military industry, computer, media, communication, banking, non-banking finance, automobile, and mechanical equipment, a total of 28 categories. The table below shows the historical market chart of Shenwan Index Level 1 industries on February 5, 2021.

##### 4.3.3.1 Historical Market Chart of Shenwan Index Level 1 Industries on February 5, 2021

| Index Code | Index Name         | Release Date     | Open Index | High Index | Low Index  | Close Index | Volume (100 Million Hands) | Turnover (100 Million CNY) | Change (%) |
| :--------- | :----------------- | :--------------- | :--------- | :--------- | :--------- | :---------- | :------------------------- | :------------------------- | :--------- |
| 801010     | Agriculture, Forestry, Animal Husbandry and Fishery | 2021/2/5 0:00  | 4111.43    | 4271.09    | 4072.53    | 4081.81     | 15.81                      | 307.82                     | -0.3       |
| 801020     | Mining             | 2021/2/5 0:00  | 2344.62    | 2357.33    | 2288.97    | 2289.41     | 18.06                      | 115.6                      | -2.25      |
| 801030     | Chemical Industry    | 2021/2/5 0:00  | 4087.77    | 4097.59    | 3910.67    | 3910.67     | 55.78                      | 778.85                     | -3.95      |
| 801040     | Steel              | 2021/2/5 0:00  | 2253.78    | 2268.17    | 2243.48    | 2250.81     | 11.61                      | 48.39                      | -1.02      |
| 801050     | Nonferrous Metals    | 2021/2/5 0:00  | 4212.1     | 4250.59    | 4035.99    | 4036.74     | 45.41                      | 593.92                     | -4.43      |
| 801080     | Electronic Components | 2021/2/5 0:00  | 4694.8     | 4694.8     | 4561.95    | 4561.95     | 52.67                      | 850.79                     | -2.78      |
| 801110     | Household Appliances | 2021/2/5 0:00  | 10033.82   | 10171.26   | 9968.93    | 10096.83    | 8.55                       | 149.18                     | 0.83       |
| 801120     | Food and Beverage    | 2021/2/5 0:00  | 30876.33   | 31545.02   | 30649.57   | 30931.69    | 11.32                      | 657.11                     | 0.47       |
| 801130     | Textile and Apparel  | 2021/2/5 0:00  | 1614.48    | 1633.89    | 1604.68    | 1607.63     | 6.28                       | 57.47                      | -0.39      |
| 801140     | Light Industry Manufacturing | 2021/2/5 0:00 | 2782.07    | 2791.88    | 2735.48    | 2737.24     | 15.28                      | 176.16                     | -1.35      |
| ...        | ...                | ...              | ...        | ...        | ...        | ...         | ...                        | ...                        | ...        |

**Data Source**: Shenwan Hongyuan

The table below is partial data of the original factors. After sequentially going through the four factor cleaning steps of missing value imputation, factor outlier removal, factor standardization, and factor neutralization mentioned above, partial data of the cleaned factors are obtained as shown in the table.

##### 4.3.3.2 Original Factor Data

| trade_date | sec_code    | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
| :----------- | :------------ | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :-- |
| 2005-01-04 | 600874.SH   | 0.001684 | NaN      | 9.445412 | 9.445412 | 9.445408 | -1.00    | NaN      | 12651.124023 | ... |
| 2005-01-04 | 000411.SZ   | 0.021073 | NaN      | 5.971262 | 5.971262 | 5.971313 | 0.38     | NaN      | 392.124298 | ... |
| 2005-01-04 | 000979.SZ   | 0.021207 | NaN      | 6.768918 | 6.768918 | 6.768815 | -1.45    | NaN      | 870.587585 | ... |
| 2005-01-04 | 000498.SZ   | 0.030220 | NaN      | 8.852752 | 8.852752 | 8.852755 | 0.55     | NaN      | 6994.011719 | ... |
| 2005-01-04 | 600631.SH   | 0.015699 | NaN      | 9.589897 | 9.589897 | 9.589889 | -1.70    | NaN      | 14616.806641 | ... |

##### 4.3.3.3 Cleaned Factor Data

| sec_code  | trade_date | ret      | factor_0 | factor_1 | factor_2 | factor_3 | factor_4 | factor_5 | factor_6 | ... |
| :---------- | :----------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :-- |
| 000001.SZ | 2005-01-04 | -1.58653 | 0.01545  | 1.38306  | 1.38306  | 1.38306  | 0.13392  | 0.01545  | 1.38564  | ... |
| 000002.SZ | 2005-01-04 | 1.36761  | -0.44814 | 1.69728  | 1.69728  | 1.69728  | 1.04567  | -0.44814 | 1.69728  | ... |
| 000004.SZ | 2005-01-04 | 0.32966  | -1.41654 | -0.13907 | -0.13907 | -0.13907 | -0.34769 | -1.41654 | -0.13650  | ... |
| 000005.SZ | 2005-01-04 | 0.61297  | -1.13066 | 1.05339  | 1.05339  | 1.05339  | -1.20020 | -1.13066 | 1.05597  | ... |
| 000006.SZ | 2005-01-04 | -0.35542 | 1.67667  | -0.07726 | -0.07726 | -0.07726 | 1.55820  | 1.67667  | -0.07469  | ... |

### 4.4 Factor Selection Based on LightGBM

#### 4.4.1 GBDT
Gradient Boosting Decision Tree (GBDT), proposed by [Friedman (2001)](https://www.jstor.org/stable/2699986)$^{[20]}$, is an iterative regression decision tree. Its main idea is to optimize the model by gradually adding weak classifiers (usually decision trees), so that the overall model can minimize the loss function. The GBDT model can be expressed as:

$$
\hat{y} = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})
$$

Where:
- \( M \) is the number of iterations,
- \( \gamma_m \) is the weight of the \( m \)-th weak classifier,
- \( h_m(\mathbf{x}) \) is the \( m \)-th decision tree model.

The training process of GBDT minimizes the loss function by gradually fitting the negative gradient direction. The specific update formula is:

$$
\gamma_m = \arg\min_\gamma \sum_{i=1}^{N} L\left(y_i, \hat{y}_{i}^{(m-1)} + \gamma h_m(\mathbf{x}_i)\right)
$$

Where, \( L \) is the loss function, \( y_i \) is the true value, and \( \hat{y}_{i}^{(m-1)} \) is the predicted value after the \( (m-1) \)-th iteration.

#### 4.4.2 LightGBM
[Light Gradient Boosting Machine (LightGBM)](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)$^{[21]}$ is an efficient framework for implementing the GBDT algorithm, initially developed by Microsoft as a free and open-source distributed gradient boosting framework. LightGBM is based on decision tree algorithms and is widely used in ranking, classification, and other machine learning tasks. Its development focuses on performance and scalability. Its main advantages include high-efficiency parallel training, faster training speed, lower memory consumption, better accuracy, and support for distributed computing and fast processing of massive data$^{[22]}$.

The core algorithm of LightGBM is based on the following optimization objective:

$$
L = \sum_{i=1}^{N} l(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(h_m)
$$

Where, \( l \) is the loss function, and \( \Omega \) is the regularization term, used to control model complexity, usually expressed as:

$$
\Omega(h_m) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

Here, \( T \) is the number of leaves in the tree, \( w_j \) is the weight of the \( j \)-th leaf, and \( \gamma \) and \( \lambda \) are regularization parameters.

LightGBM uses technologies such as Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), which significantly improve training efficiency and model performance.

In this study, the loss function used during training is Mean Squared Error (MSE), which is defined as:

$$
L(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where, \( y \) is the true return rate, \( \hat{y} \) is the return rate predicted by the model, and \( N \) is the number of samples.

#### 4.4.3 Algorithm Flow
The specific algorithm flow in this section is as follows:

1.  **Data Preparation**: Use one year's worth of 50 factor data for each stock (A-share market-wide data) and historical future one-month returns as features.
2.  **Model Training**: Use Grid Search to optimize the hyperparameters of the LightGBM model and train the model to predict the future one-month return rate. The model training flow is shown in Fig. 4.12.

    $$
    \text{Parameter Optimization:} \quad \theta^* = \arg\min_\theta \sum_{i=1}^{N} L(y_i, \hat{y}_i(\theta))
    $$

    Where, \( \theta \) represents the set of model hyperparameters, and \( \theta^* \) is the optimal parameter.

3.  **Factor Importance Calculation**: Use LightGBM's `feature_importances_` method to calculate the feature importance of each factor. Feature importance is mainly measured by two indicators:
    -   **Split**: The number of times the feature is used for splitting in all trees.
    -   **Gain**: The total gain brought by the feature in all splits (i.e., the amount of reduction in the loss function).

    The feature importance of a factor can be expressed as:

    $$
    \text{Importance}_{\text{split}}(f) = \sum_{m=1}^{M} \sum_{j=1}^{T_m} \mathbb{I}(f \text{ is used for splitting the } j \text{-th leaf node})
    $$

    $$
    \text{Importance}_{\text{gain}}(f) = \sum_{m=1}^{M} \sum_{j=1}^{T_m} \Delta L_{m,j} \cdot \mathbb{I}(f \text{ is used for splitting the } j \text{-th leaf node})
    $$

    Where, \( \mathbb{I} \) is the indicator function, and \( \Delta L_{m,j} \) is the reduction in loss brought by factor \( f \) in the \( j \)-th split of the \( m \)-th tree.

4.  **Factor Screening**: Sort according to the factor importance calculated by the model, and select the top ten factors with the highest importance as the factors used in this cross-sectional analysis. The importance of the selected factors is shown in Table 4.4.4.

#### 4.4.4 Partial Ranking of Selected Factor Importance

| importance | feature_name | trade_date  |
| :--------- | :----------- | :---------- |
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

#### 4.4.5 Code Implementation Snippet

The following is a code snippet used in the training process for factor selection.

{{< collapse summary="feature_choice" openByDefault=false >}}
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
{{< /collapse >}}

Through the above process, LightGBM is used to efficiently screen out the factors that have the greatest impact on predicting future returns, thereby improving the predictive ability and interpretability of the model.

### 4.5 Factor Combination Based on BiLSTM

This section uses BiLSTM for factor combination. The specific principle of BiLSTM has been introduced in Chapter 2, and will not be repeated here. First, let's introduce the specific network structure of the model. The network structure of BiLSTM set in this paper through a large number of repeated experiments is shown in Table 4.5.1. The default tanh and linear activation functions of recurrent neural networks are used between layers. Dropout is added to prevent overfitting, but if Dropout uses an excessively large dropout rate, underfitting will occur. Therefore, the dropout rate of Dropout is set to 0.01. The number of neurons in the BiLSTM recurrent layer of the final model is 100. A BiLSTM layer and three fully connected layers are used, and a Dropout is set between the BiLSTM layer and the first fully connected layer.

#### 4.5.1 BiLSTM Network Structure

| Layer(type)                   | Output Shape | Param# |
| :------------------------------- | :----------- | :----- |
| bidirectional_1 (Bidirection) | (None, 100)  | 24400  |
| dropout_1 (Dropout)           | (None, 100)  | 0      |
| dense_1 (Dense)               | (None, 256)  | 25856  |
| dropout_2 (Dropout)           | (None, 256)  | 0      |
| dense_2 (Dense)               | (None, 64)   | 16448  |
| dense_3 (Dense)               | (None, 1)    | 0      |

**Total params**: 66,769  
**Trainable params**: 66,769  
**Non-trainable params**: 0

Because the amount of data used in this experiment is large, `epochs=400` and `batch_size=1024` are selected. The loss function of the model is Mean Squared Error (MSE). The optimizer used is Stochastic Gradient Descent (SGD). Stochastic gradient descent has three advantages over gradient descent (GD): it can more effectively use information when information is redundant, and the early iteration effect is excellent, which is suitable for processing large-sample data $^{[23]}$. Since the amount of training data in this experiment is large, if SGD is used, only one sample is used for iteration each time, and the training speed is very fast, which can greatly reduce the time spent on training. The default values in its `keras` package are used, i.e., `lr=0.01`, `momentum=0.0`, `decay=0.0`, and `nesterov=False`.

**Parameter Explanation:**

-   `lr`: Learning rate
-   `momentum`: Momentum parameter
-   `decay`: Learning rate decay value after each update
-   `nesterov`: Determine whether to use Nesterov momentum

#### 4.5.2 Algorithm Flow
The specific algorithm flow in this section is as follows:

1.  Use A-share market-wide data of 10 factors (factors selected by LightGBM) and historical future one-month returns for each stock for one year as features.
2.  Take the future one-month return rate of each stock per year as the prediction target, and use BiLSTM for training, as shown in Fig. 12.

{{< figure
    src="Rolling_Window.png"
    caption="Fig. 12. Rolling Window"
    align="center"
>}}

3.  The real-time factor data of out-of-sample data for one month is passed through the trained BiLSTM model to obtain the real-time expected return rate of each stock for the next month. The return rate is shown in Table 4.11.

#### 4.5.3 Partial Stock Predicted Return Rate Table

| sec_code  | trade_date | y_hat      |
| :---------- | :----------- | :----------- |
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

#### 4.5.4 Code Implementation Snippet

The following is a code snippet used in the training process for building the BiLSTM training network.

{{< collapse summary="build_net_blstm" openByDefault=false >}}
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
{{< /collapse >}}

### 4.6 Quantitative Strategy and Strategy Backtesting

#### 4.6.1 Backtesting Metrics

First, let's introduce some common backtesting metrics for strategies. Evaluation metrics include Total Rate of Return, Annualized Rate of Return, Annualized volatility, Sharpe ratio, Maximum Drawdown (MDD), Annualized turnover rate, and Annualized transaction cost rate. It is assumed that the stock market is open for 252 days a year, the risk-free rate is defaulted to 0.035, and the commission fee is defaulted to 0.002.

1.  **Total Rate of Return**: Under the same other indicators, the larger the cumulative rate of return, the better the strategy, and the more it can bring greater returns. The formula is as follows:

$$
\text{Total Rate of Return} = r_{p} = \frac{P_{1} - P_{0}}{P_{0}}
$$

$P_{1}$: Total value of final stocks and cash  
$P_{0}$: Total value of initial stocks and cash

2.  **Annualized Rate of Return**: It is to convert the cumulative total rate of return into a geometric average rate of return on an annual basis. Under the same other indicators, the larger the annualized rate of return, the better the strategy. The formula is as follows:

$$
\text{Annualized Rate of Return} = R_{p} = \left(1 + r_{p}\right)^{\frac{252}{t}} - 1
$$

$r_{p}$: Cumulative rate of return  
$t$: Number of days the investment strategy is executed

3.  **Annualized volatility**: Defined as the standard deviation of the logarithmic value of the annual return rate of the object asset. Annualized volatility is used to measure the risk of a strategy. The greater the volatility, the higher the risk of the strategy. The formula is as follows:

$$
\begin{aligned}
\text{Annualized volatility} = \sigma_{p} &= \sqrt{\frac{252}{t-1} \sum_{i=1}^{t}\left(r_{d} - \bar{r}_{d}\right)^{2}} \\
\bar{r}_{d} &= \frac{1}{t} \sum_{i=1}^{t} r_{d_{i}}
\end{aligned}
$$

$r_{d_{i}}$: Daily return rate on the $i$-th day  
$\bar{r}_{d}$: Average daily return rate  
$t$: Number of days the investment strategy is executed

4.  **Sharpe ratio**: Proposed by [Sharpe (1966)](https://doi.org/10.2307/2328485)$^{[24]}$. It represents the excess return obtained by investors for bearing an extra unit of risk$^{[25]}$. Here is the calculation formula for the annualized Sharpe ratio:

$$
S = \frac{R_{p} - R_{f}}{\sigma_{p}}
$$

$R_{p}$: Annualized rate of return  
$R_{f}$: Risk-free rate of return  
$\sigma_{p}$: Annualized volatility

5.  **Maximum Drawdown (MDD)**: Indicates the maximum value of the return rate drawdown when the total value of stocks and cash of our strategy portfolio reaches the lowest point during the operation period. Maximum drawdown is used to measure the most extreme possible loss situation of the strategy.

$$
MDD = \frac{\max \left(V_{x} - V_{y}\right)}{V_{x}}
$$

$V_{x}$ and $V_{y}$ are the total value of stocks and cash of the strategy portfolio on day $x$ and day $y$ respectively, and $x < y$.

6.  **Annualized turnover rate**: Used to measure the frequency of buying and selling stocks in the investment portfolio. The larger the value, the more frequent the portfolio turnover and the greater the transaction cost.

$$
\text{change} = \frac{N \times 252}{t}
$$

$t$: Number of days the investment strategy is executed  
$N$: Total number of buy and sell transactions

7.  **Annualized transaction cost rate**: Used to measure the transaction cost of the investment portfolio strategy. The larger the value, the higher the transaction cost.

$$
c = \left(1 + \text{commison}\right)^{\text{change}} - 1
$$

change: Annualized turnover rate  
commison: Commission fee

#### 4.6.2 Strategy and Backtesting Results

The quantitative trading strategy in this paper adopts position switching every month (i.e., the rebalancing period is 28 trading days). Each time, the strategy adopts an equal-weight stock holding method to buy the 25 stocks with the highest expected return rate predicted by BiLSTM and sell the originally held stocks. The backtesting time and rules in this paper are as follows:

1.  **Backtesting Time**: From January 2012 to October 2020.
2.  **Backtesting Stock Pool**: All A-shares, excluding Special Treatment (ST) stocks.
3.  **Transaction Fee**: A brokerage commission of 0.2% is paid when buying, and a brokerage commission of 0.2% is paid when selling. If the commission for a single transaction is less than 5 CNY, the brokerage charges 5 CNY.
4.  **Buying and Selling Rules**: Stocks that hit the upper limit on the opening day cannot be bought, and stocks that hit the lower limit cannot be sold.

##### 4.6.2.1 Strategy Backtesting Results

|                 | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown | Annualized Turnover Rate | Annualized Transaction Cost Rate |
| :---------------- | :-------------- | :---------------- | :------------------ | :----------- | :----------- | :----------------------- | :----------------------------- |
| **Strategy**      | 701.00%         | 29.18%            | 33.44%              | 0.77         | 51.10%       | 51.10%                   | 11.35%                         |
| **Benchmark**     | 110.40%         | 9.70%             | 26.01%              | 0.24         | 58.49%       | 58.49%                   | 0.00%                          |

{{< figure
    src="res.png"
    caption="Fig. 22. Net Profit Curve"
    align="center"
>}}

The backtesting results are shown in the table and Fig. 22 above. My strategy adopts the LightGBM-BiLSTM quantitative strategy introduced in this chapter. The benchmark uses the CSI All Share (000985). From the results above, it can be seen that the cumulative return of this strategy is 701.00%, which is much higher than the benchmark's 110.40%; the annualized return is 29.18%, which is much higher than the benchmark's 9.70%; and the Sharpe ratio is 0.77, which is higher than the benchmark's 0.24. These three backtesting indicators show that the LightGBM-BiLSTM quantitative strategy can indeed bring greater returns to investors. The annualized volatility of this strategy is 33.44%, which is greater than the benchmark's 26.01%, and the maximum drawdown is 51.10%, which is less than the benchmark's 58.49%. These two backtesting indicators show that the LightGBM-BiLSTM quantitative strategy has certain risks, especially it is difficult to resist the impact of systemic risks. The annualized turnover rate is 11.35%, and the annualized transaction cost rate is 2.29%, indicating that our strategy is not a high-frequency trading strategy and the transaction cost is small. It can be seen from the return curve chart that the return rate of the LightGBM-BiLSTM quantitative strategy in the first two years is not much different from the benchmark, and there is no special advantage. However, from around April 2015, the return rate of the LightGBM-BiLSTM quantitative strategy is significantly better than the benchmark's return rate. Overall, the return rate of this LightGBM-BiLSTM quantitative strategy is very considerable, but there are still certain risks.

## Chapter 5 Conclusion and Future Directions

### 5.1 Conclusion

This paper first introduced the research background and significance of stock price prediction and quantitative strategy research based on deep learning, and then introduced the domestic and international research status of stock price prediction and quantitative investment strategies respectively. Then, the innovations and research framework of this paper were explained. Then, in the chapter on related theoretical foundations, this paper briefly introduced the deep learning models and the development history of quantitative investment used in this paper. The basic structure, basic principles, and characteristics of the three models LSTM, GRU, and BiLSTM are mainly introduced.

Subsequently, this paper used the daily frequency data of SPD Bank and IBM, and preprocessed the data through a series of data processing processes and feature extraction. Then, the specific network structure and hyperparameter settings of the three models LSTM, GRU, and BiLSTM were introduced. Then, we used LSTM, GRU, and BiLSTM to predict the closing prices of the two stocks and compare the model evaluations. The experimental results show that for both stocks, the BiLSTM prediction effect is more accurate.

Finally, in order to further illustrate the application value of BiLSTM in finance, this paper constructed a quantitative investment model based on LightGBM-BiLSTM. Stocks in the entire A-share market and multiple factors were selected for factor cleaning, factor selection based on LightGBM, and factor combination based on LSTM. Then, we constructed a certain investment strategy and compared it with the benchmark holding CSI All Share through evaluation indicators such as cumulative return rate, annualized return rate, annualized volatility, and Sharpe ratio. Through comparison, it was found that the LightGBM-BiLSTM quantitative investment model can bring better returns, indicating the effectiveness of using deep learning to build quantitative investment strategies.

### 5.2 Future Directions

Although this paper compares the effects of LSTM, GRU, and BiLSTM models in predicting stock closing prices and achieves certain results based on the LightGBM-BiLSTM quantitative investment strategy, there are still some shortcomings in this paper's research. Combining the research results of this paper, the following research and improvements can be further carried out:

1.  **Diversification of Prediction Targets**: In terms of predicting stock prices, this paper selects the stock closing price as the prediction target. Although this result is the most intuitive, the Random Walk Hypothesis (RWH) proposed by [Bachelier (1900)](http://www.numdam.org/item/ASENS_1900_3_17__21_0/)$^{[26]}$ believes that stock prices follow a random walk and are unpredictable. Although many behavioral economists have since proved that this view is not entirely correct, it also shows that simply predicting stock closing prices is not so strong in terms of difficulty and interpretability $^{[27][28]}$. Therefore, stock volatility prediction, stock price increase/decrease judgment, and stock return rate prediction can be selected as future research directions.
2.  **Diversified Model Comparison**: In terms of predicting stock prices, this paper compares the three recurrent neural network models LSTM, GRU, and BiLSTM and shows that BiLSTM has better prediction effect, but there is still a lack of comparative research with more different models. Therefore, future in-depth research can be conducted on comparisons with Autoregressive Integrated Moving Average (ARIMA), Convolutional Neural Networks (CNNs), Deep Neural Networks (DNNs), CNN-LSTM, Transformer, TimeGPT, and other single or composite models.
3.  **Factor Diversification**: The factors used in this paper to construct quantitative investment strategies are all technical price-volume factors, and the types of factors are single. In the future, different types of factors such as financial factors, sentiment factors, and growth factors can be selected to improve the performance of the strategy. At the same time, future research can also appropriately add timing strategies to increase positions when predicting that the market will rise and reduce positions when predicting that the market will fall to earn beta (\(\beta\)) returns.
4.  **Investment Portfolio Optimization**: The factor combination process in this paper is still imperfect. In the future, quadratic programming methods can be used to optimize the investment portfolio.
5.  **High-Frequency Trading Strategy Research**: The quantitative investment strategy method in this paper adopts a low-frequency trading strategy. In the future, stock tick data can be used to study high-frequency strategies and ultra-high-frequency strategies.

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

### Reference Blog

- Colah's Blog. (2015, August 27). [*Understanding LSTM Networks.*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Citation

> **Citation**: For reprint or citation of this article, please indicate the original author and source.

**Cited as:**

> Yue Shui. (Apr 2021). Stock Price Prediction and Quantitative Strategy Based on Deep Learning.
https://syhya.github.io/posts/2021-04-21-deep-learning-stock-prediction/

Or

```bibtex
@article{syhya2021stockprediction,
  title   = "Stock Price Prediction and Quantitative Strategy Based on Deep Learning",
  author  = "Yue Shui",
  journal = "syhya.github.io",
  year    = "2021",
  month   = "Apr",
  url     = "https://syhya.github.io/posts/2021-04-21-deep-learning-stock-prediction/"
}
```
