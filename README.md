# Audit Sampling With Autoencoders
Seeing financial transactions in lower dimensions with neural networks

<p align="center">
<img src="images/marbles.jpg" width=600>
<p/>

## Table of Contents
1. [File Descriptions](#files)
2. [Supporting Packages](#packages)
3. [How To Use This Repository](#howto)
4. [Project Motivation](#motivation)
5. [About The Dataset](#data)
6. [Results](#results)
7. [Acknowledgements](#acknowledgements)
8. [Licence & copyright](#license)

## File Descriptions <a name="files"></a>
| File | Description |
| :--- | :--- |
| data/city_payments_fy2017.csv | features: dept, trans amount, date, purchased item, etc.|
| ASWA.ipynb | jupyter notebook used to develop analysis |
| preprocessing.py | module for ETL, prepares data for neural network |
| models.py | module with autoencoder classes and methods for analysis |

## Supporting Packages <a name="packages"></a>
In addition to the standard python library, this analysis utilizes the following packages:
- [Datapane](https://datapane.com/)
- [NumPy](https://numpy.org/?msclkid=8a02e767b93111ecae80e39be02e750a)
- [pandas](https://pandas.pydata.org/?msclkid=96098534b93111ec801d615628da32cd)
- [PyTorch](https://pytorch.org/)
- [Plotly](https://plotly.com/)
- [scikit-learn](https://scikit-learn.org/stable/)

Please see `requirements.txt` for a complete list of packages and dependencies used in the making of this project

## How To Use This Repository <a name="howto"></a>
1. Download and unzip this repository to your local machine.
2. Navigate to this directory and open the command line. For the purposes of running the scripts, this will be the root directory.
3. Create a virtual environment to store the supporting packages

        python -m venv ./venv

4. Activate the virtual environment

        venv\scripts\activate

5. Install the supporting packages from the requirements.txt file

        pip install -r requirements.txt
        
6. To run the ETL pipeline that cleans data and pickles it, type the following in the command line:
        
        python preprocessing.py data/city_payments_fy2017.csv

7. To train a traditional autoencoder and save the model locally, type the following in the command line:

        python models.py data/philly_payments_clean ae 5
       
Note: This is provided as an example, you can also chooose "vae" and a number of epochs other than 5. 
When training is complete an html file is generated providing a visualization of the embedded transaction data.

## Project Motivation <a name="motivation"></a>
Auditing standards require the assessment of the underlying transactions that comprise the financial statements to detect errors or fraud that would result in material misstatement. The accounting profession has developed a framework for addressing this requirement, known as the Audit Risk Model.

The Audit Risk Model defines audit risk as the combination of inherent risk, control risk and detection risk:

<p align="center">
<img src="images/audit risk model.png" width=600>
</p>

Detection risk is composed of sampling risk and non-sampling risk:

<p align="center">
<img src="images/detection risk decomposition.png" width=600>
</p>

Sampling risk is defined as the risk that the auditor's conclusion based on the sample would be different had the entire population been tested. In other words, it is the risk that the sample is not representative of the population and does not provide sufficient appropriate audit evidence to detect material misstatements.

There are a variety of sampling methods used by auditors. Random sampling is based on each member of the population having an equal chance of being selected. Stratified sampling subdivides the population into homogenous groups from which to make selections. Monetary unit sampling treats each dollar amount from the population as the sampling unit and selects items when a cumulative total meets or exceeds a predefined sampling interval when cycling through the population.

**Autoencoders** offer an alternative method for addressing sampling risk. An autoencoder is a neural network that learns to encode data into lower dimensions and decode it back into higher dimensions. The resulting model provides a low-dimensional representation of the data, disentangling it in a way that reveals something about its fundamental structure. Auditors can model transactions in this way and select from low-dimensional clusters. They can also identify anomalous transactions based on how much they deviate from other transactions in this latent space.

In this demonstration, we consider the traditional autoencoder:

<p align="center">
<img src="images/ae.png" width=600>
<p/>

as well as a **variational autoencoder**:

<p align="center>
<img src="images/vae.png" width=600>
<p/>

There is an important distinction between a variational autoencoder and  a traditional autoencoder. There are generally two output layers of the encoder that represent the means and standard deviations of the underlying distributions of the data. Further, the latent matrix Z is determined by sampling from a Gaussian distribution parameterized by the learned means and standard deviations of the latent space.

Now, it is not actually feasible to perform backpropagation with the configuration above because the sampling operation is not differentiable. In practice, we instead sample a random matrix epsilon from a normal distribution and scale the latent standard deviations by epsilon by applying the element-wise product. We then add the result to the latent means to obtain the latent embedding Z.

<p align="center>
<img src="images/vae2.png" width=600>
<p/>

Structuring the network in this way allows for both stochastic sampling and differentiation with respect to the latent means and standard deviations.

## About The Dataset <a name="data"></a>
To demonstrate how autoencoders work, we analyze the [City of Philadelphia payments data](https://www.phila.gov/2019-03-29-philadelphias-initial-release-of-city-payments-data/). It is one of two datasets used in [Schreyer et al (2020)](https://arxiv.org/pdf/2008.02528v1.pdf) and consists of nearly a quarter-million payments from 58 city offices, departments, boards, and commissions. It covers the City's fiscal year 2017 (July 2016  through June 2017) and represents nearly $4.2 billion in payments during that period.

## Results <a name="results"></a>

<p align="center">
<img src="images/loss_plot.png" width=600>

<img src="images/projection_plot.png" width=600>
<p/>

To read more about this project, check out this [Medium post](https://medium.com/@zacharywolinsky/audit-sampling-with-autoencoders-90ddd54fd1c).

## Acknowledgements <a name="acknowledgements"></a>
This project is largely inspired by a paper published in 2020 by Marco Schreyer, Timur Sattarov, Anita Gierbl, Bernd Reimer, and Damian Borth, entitled [Learning Sampling in Financial Statement Audits using Vector Quantised Autoencoder Neural Networks](https://arxiv.org/pdf/2008.02528.pdf), as well as an excellent demonstration of autoencoders by Alexander Van de Kleut, entitled [Variational Autoencoders (VAE) with PyTorch](https://avandekleut.github.io/vae/).

## License & copyright <a name="license"></a>
© Zachary Wolinsky 2022

Licensed under the [MIT License](LICENSE.txt)
