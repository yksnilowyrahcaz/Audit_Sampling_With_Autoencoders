import sys
import time
import torch
import itertools
import pandas as pd
torch.manual_seed(0)
import torch.nn as nn
import datapane as dp
import plotly.express as px
import torch.nn.functional as F
import plotly.graph_objects as go
from preprocessing import Preprocessor

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# ...............................SHARED FUNCTIONS..................................#
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#  

def fit(self, data, epochs=1, gamma=1e-3):
    '''
    Iteratively train the autoencoder,
    optimizing over parameter space by minimizing
    mean squared error (MSE). Collect training
    and validation losses for model evaluation.

    Args: 
        data: 
            torch.utils.data.DataLoader object 
            that contains the dataset of interest

        epochs:
            number of times to batch cycle through 
            all of the data being learned from. 
            Default: 1

        gamma:
            weight applied to the Kullback–Leibler divergence
            term of the loss function. 
            Default: 1e-3 (0.001)
    '''
    self.test_loss = []
    self.train_loss = []
    test_set = itertools.cycle(data.test_set)
    opt = torch.optim.Adam(self.parameters())
    for epoch in range(epochs):
        start_time = time.time()
        for x_train, y_train in data.train_set:
            self.train()
            x_train = x_train.to(device)
            opt.zero_grad()
            x_train_pred = self.decode(self.encode(x_train))
            loss = ((x_train - x_train_pred)**2).sum() + gamma*self.kl
            self.train_loss.append(loss.item())
            loss.backward()
            opt.step()
            with torch.no_grad():
                self.eval()
                x_test = next(test_set)[0].to(device)
                x_test_pred = self.decode(self.encode(x_test))
                self.test_loss.append(
                    (((x_test - x_test_pred)**2).sum() + gamma*self.kl).item()
                ) 
        print(f'epoch: [{epoch+1}/{epochs}] | ' +
              f'loss: {round(loss.item(), 4)} | ' +
              f'elapsed time: {round((time.time() - start_time)/60, 2)} minutes')

def plot_loss(self):
    '''
    Visualize the reconstruction loss for 
    each batch observed during training,
    computed on the training and test sets.
    
    Returns:
        fig:
            plotly.express graph object
    '''
    fig = px.scatter()
    
    fig.add_trace(
        go.Scatter(
            y=self.train_loss,
            line=dict(color='forestgreen'),
            showlegend=True,
            name='Training Loss'
        )
    )
        
    fig.add_trace(
        go.Scatter(
            y=self.test_loss,
            line=dict(color='dodgerblue'),
            showlegend=True,
            name='Testing Loss'
        )
    )

    fig.update_traces(mode='lines')

    fig.update_traces(hovertemplate='<br>'.join([
        'Batch Number: %{x}',
        'Loss: %{y:,.4f}'
    ]))

    fig.update_xaxes(title_text='Batch Number',
                     gridcolor='lightgray',
                     showgrid=True, 
                     gridwidth=1, 
                     type='log')

    fig.update_yaxes(title_text='Loss',
                     gridcolor='lightgray',
                     showgrid=True, 
                     gridwidth=1,
                     type='log')

    fig.update_layout(title_font_size=20,
                      title_text='Reconstruction Loss' + \
                      '<br><sup>Log-Log Scaled</sup>',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      hoverlabel=dict(
                          bgcolor='ivory',
                          font_size=16,
                          font_family='Rockwell'),
                      font=dict(family='Rockwell', 
                                color='navy',
                                size=16), 
                      legend=dict(
                          bgcolor='white',
                          yanchor='top',
                          y=0.99,
                          xanchor='right',
                          x=0.99),
                      title_x=0.5)
    return fig

def plot_projection(self, data, which='train_set', 
                    num_batches=10, gamma=1e-3):
    '''
    Visualize the encoded transaction data, color coded by the features, 
    excluding day, month and year, which are isotropically distributed.
    
    Args: 
        data: 
            scripts.processing.Preprocessor object 
            that contains the datasets of interest,
            namely data.train_set and data.test_set

        which:
            str, 'train_set' plots the encoded training
            data, else the encoded test data in plotted.
            Default: 'train_set'

        num_batches:
            number of batches of 128 data points to
            project on two latent dimenstions 
            z.1 and z.2. Default: 10
            
        gamma:
            weight applied to the Kullback–Leibler divergence
            term of the loss function. Used in this context to
            compute the reconstruction loss for individual
            transactions to highlight outliers. 
            Default: 1e-3 (0.001)
    
    Returns:
        fig:
            plotly.express graph object
    '''
    self.scores = torch.empty((0, 1))
    self.projection = torch.empty((0, self.latent_dims))
    self.labels = torch.empty((0, data.Y.shape[1]))
    for i, (x, y) in enumerate(data.train_set 
                               if which == 'train_set' 
                               else data.test_set):
        z = self.encode(x.to(device))
        x_pred = self.decode(z).detach()
        z = z.to('cpu').detach()
        scores = ((x - x_pred)**2).sum(axis=1) + gamma*self.kl.detach()
        self.labels = torch.vstack((self.labels, y))
        self.projection = torch.vstack((self.projection, z))
        self.scores = torch.vstack((self.scores, scores.reshape(-1,1)))                   
        if i == num_batches - 1:
            self.labels = torch.hstack((self.labels, self.scores))
            break

    data_ = pd.concat([
        data.recover_labels(self.labels[:,:-1]), 
        pd.Series(self.labels[:,-1].numpy())
    ], axis=1)

    fig = px.scatter()

    fig.add_trace(
        go.Scatter(
            x=self.projection[:,0], 
            y=self.projection[:,1],
            mode='markers',
            marker=dict(size=10,
                        color=self.labels[:,5],
                        colorscale=px.colors.qualitative.Set3,
                        line=dict(width=0.5,
                                  color='DarkSlateGrey'),
                       ),
            showlegend=False,
            customdata=data_,
            name=''
        )
    )

    fig.update_traces(
        selector=dict(mode='markers'),
        hovertemplate='<br>'.join([
            'Transaction Amount: %{customdata[12]:$,.2f}'+\
            ', Sign: %{customdata[13]}',
            'Document Number: %{customdata[0]}',
            'Department Title: %{customdata[1]}',
            'Character Title: %{customdata[2]}',
            'Sub Object Title: %{customdata[3]}',
            'Vendor Name: %{customdata[4]}',
            'Doc Ref No. Prefix Definition: %{customdata[5]}',
            'Contract Description: %{customdata[6]}',
            'Payment Method: %{customdata[7]}',
            'Date: %{customdata[8]}, %{customdata[10]}'+\
            ' %{customdata[9]}, %{customdata[11]}', 
            'Reconstruction Loss: %{customdata[14]:,.4f}'
            ])
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family='Rockwell', 
            size=16, 
            color='navy'),
        hoverlabel=dict(
            font_size=14,
            font_family='Rockwell'),
        title_text='Low-dimensional projection of' +\
        '<br>the Philadelphia payments data',
        title_font_size=18,
        title_y=.92,
        title_x=.95,
        xaxis={'title':'Z.1'},
        yaxis={'title':'Z.2'}
    )

    fig.update_xaxes(gridcolor='lightgray',
                     showgrid=True, 
                     gridwidth=1)

    fig.update_yaxes(gridcolor='lightgray',
                     showgrid=True, 
                     gridwidth=1)

    buttons = [] 
    # Note: index 13 because data.features does not 
    # include documet_no feature; self.labels does
    data.feature_names[13] = 'reconstruction_loss'                               
    for index, feature in data.feature_names.items():
        if index not in [8,9,10]: # exclude day, month, year (isotropic)
            args = dict(size=10,
                        color=self.labels[:,index+1],
                        colorscale=fig.data[0].marker.colorscale 
                        if index != 13 else '',
                        line=dict(width=0.5,
                                  color='DarkSlateGrey')
                       )
              
            #create a button object to select a feature
            button = dict(label=feature,
                          method='update',
                          args=[{'marker':args}])

            #add the button to our list of buttons
            buttons.append(button)

    fig.update_layout(
        updatemenus=[dict(
            active=4,
            font={'size':12},
            bgcolor='white',
            type='dropdown',
            buttons=buttons,
            xanchor='left',
            yanchor='bottom',
            y=1,
            x=0,
        )], 
    )

    return fig

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# .............................AUTOENCODER CLASSES.................................#
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#     
    
class AE(nn.Module):
    '''
    Autoencoder class to learn a latent representation of the data.
    Uses adaptive moment estimation (Adam) to optimize the squared 
    reconstruction error over the parameter space.
    
    Args:
        input_dims:
            int, number of dimensions of the input space
        
        latent_dims:
            int, number of dimensions in the latent space
        
        output_dims:
            int, number of dimensions in the output space
            
        batch_norm:
            bool, True for implementing batch normalization
            before the two intermediary hidden layers of the
            autoencoder network. Default: True.     
    '''
  
    def __init__(self, input_dims, latent_dims, 
                 output_dims, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.linear1 = nn.Linear(input_dims, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(latent_dims, 512)
        self.linear4 = nn.Linear(512, output_dims)
        # dummy attribute for fit method
        self.kl = torch.tensor([0])
        
    def encode(self, x):
        '''
        Forward pass through the encoder network
        
        Args: 
            x, torch.FloatTensor of the data in the
            input space
            
        Returns:
            z, torch.FloatTensor object of encoded data
        '''
        x = F.leaky_relu(
            self.bn1(self.linear1(x)) 
            if self.batch_norm 
            else self.linear1(x))
        z = self.linear2(x)
        return z
    
    def decode(self, z):
        '''
        Forward pass through the decoder network
        
        Args: 
            z, torch.FloatTensor object of encoded data
            
        Returns:
            x_pred, torch.FloatTensor object of decoded data
        '''
        z = F.leaky_relu(
            self.bn2(self.linear3(z)) 
            if self.batch_norm 
            else self.linear3(z))
        x_pred = torch.sigmoid(self.linear4(z))
        return x_pred

    fit = fit      
    plot_loss = plot_loss
    plot_projection = plot_projection
    
class VAE(nn.Module):
    '''
    Variational autoencoder class to learn latent representation of the data.
    Uses modified loss function that minimizes the squared reconstruction error
    plus the Kullback–Leibler divergence between the modeled distribution of the
    data and a normal distribution. 
    
    Args:
        input_dims:
            int, number of dimensions of the input space

        latent_dims:
            int, number of dimensions in the latent space

        output_dims:
            int, number of dimensions in the output space

        batch_norm:
            bool, True for implementing batch normalization
            before the two intermediary hidden layers of the
            autoencoder network. Default: True.  
    '''
    def __init__(self, input_dims, latent_dims, 
                 output_dims, batch_norm=True):
        super().__init__()
        self.batch_norm = batch_norm
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.linear1 = nn.Linear(input_dims, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)
        self.linear4 = nn.Linear(latent_dims, 512)
        self.linear5 = nn.Linear(512, output_dims)
        self.N = torch.distributions.Normal(0, 1)
        
        # can use for sampling on GPU:
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        
    def encode(self, x):
        '''
        Forward pass through the encoder network, with two encoding layers 
        representing distributions of the mean and standard deviations of 
        the overall distribution of the data being modeled. Includes a sampling 
        step that samples a mean and standard deviation from each of these two
        encoding layers to obtain a single encoding layer z. Includes the 
        computation of the Kullback–Leibler divergence for the loss function.
        
        Args: 
            x, torch.FloatTensor of the data in the
            input space
            
        Returns:
            z, torch.FloatTensor object of the sampled
            mean and standard deviations from the two
            encoding layers
        '''
        x = F.leaky_relu(
            self.bn1(self.linear1(x)) 
            if self.batch_norm 
            else self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z
    
    def decode(self, z):
        '''
        Forward pass through the decoder network
        
        Args: 
            z, torch.FloatTensor object of encoded data
            
        Returns:
            x_pred, torch.FloatTensor object of decoded data
        '''
        z = F.leaky_relu(
            self.bn2(self.linear4(z)) 
            if self.batch_norm 
            else self.linear4(z))
        x_pred = torch.sigmoid(self.linear5(z))
        return x_pred
    
    fit = fit      
    plot_loss = plot_loss
    plot_projection = plot_projection
                
def main():
    if len(sys.argv) == 4:
        start_time = time.time()
        philly_payments_clean, model_choice, epochs = sys.argv[1:]
        
        print(f'Loading dataset from: {philly_payments_clean}')
        data = torch.load(philly_payments_clean)
        dims = data.X_train.shape[1]
        
        print('Training the autoencoder...')
        model = AE(dims,2,dims) if model_choice == 'ae' else VAE(dims,2,dims)
        model.fit(data, epochs=int(epochs))
        
        print('Model training complete! Saving model...')
        torch.save(model, 'data/trained_model')
        
        print('Generating plot of projection...')
        report = dp.Report(dp.Plot(model.plot_projection(data)))
        report.save('data/model_plot.html', open=True)
              
        print(f'Total Time: {round((time.time() - start_time)/60, 2)} minutes')
    
    else:
        print(
            '''
            Please provide the following arguments when calling models.py:
            arg1: filepath of models.py
            arg2: filepath of the preprocessed data
            arg3: choice of model, ae for autoencoder or vae for variational autoencoder
            arg4: integer, number of epochs over which to train the model

            Example: python models.py data/philly_payments_clean ae 5
            '''
        )
            
if __name__ == '__main__':
    main()