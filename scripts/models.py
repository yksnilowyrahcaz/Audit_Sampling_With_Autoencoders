import sys
import time
import torch
torch.manual_seed(1729)
import torch.nn as nn
import datapane as dp
import plotly.express as px
import torch.nn.functional as F
import plotly.graph_objects as go
from preprocessing import Preprocessor

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# ...........................VISUALIZATION FUNCTIONS...............................#
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#  

def plot_loss(self):
    '''
    Visualize the reconstruction loss for 
    each batch observed during training
    
    Returns:
        fig:
            plotly.express graph object
    '''
    fig = px.line(y=[x.detach() for x in self.losses], 
                  log_x=True, log_y=True, 
                  labels={'x':'Batch Number', 
                          'y':'Squared Error'})

    fig.update_traces(mode='markers+lines')

    fig.update_traces(marker_line_width=1, 
                      marker_line_color='black',
                      marker=dict(color='lightgreen'),
                      hovertemplate='<br>'.join([
                          'Batch Number: %{x}',
                          'Squared Error: %{y:,.0f}']))

    fig.update_xaxes(gridcolor='lightgray',
                     showgrid=True, 
                     gridwidth=1)

    fig.update_yaxes(gridcolor='lightgray',
                     showgrid=True, 
                     gridwidth=1)

    fig.update_layout(title_font_size=20,
                      title_text='Training Reconstruction Loss',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      hoverlabel=dict(
                          bgcolor='ivory',
                          font_size=16,
                          font_family='Rockwell'),
                      font=dict(family='Rockwell', 
                                color='navy',
                                size=16), 
                      title_x=0.5)
    return fig

def plot_projection(self, data, num_batches=10):
    '''
    Provides a way to visualize the encoded transaction data
    color coded by the the features, excluding day, month and year,
    which are empirically isotropic in their distribution.
    
    Args: 
        data: 
            torch.utils.data.DataLoader object 
            that contains the dataset of interest

        num_features:
            number of features that comprise the dataset

        num_batches:
            number of batches of 128 data points to
                    project on two latent dimenstions z.1 and z.2
    
    Returns:
        fig:
            plotly.express graph object
    '''
    self.scores = torch.empty((0, 1))
    self.projection = torch.empty((0, self.latent_dims))
    self.labels = torch.empty((0, data.preprocessed.shape[1]))
    for i, (x, y) in enumerate(data.dataset):
        z = self.encode(x.to(device))
        x_hat = self.decode(z).detach()
        z = z.to('cpu').detach()
        self.scores = torch.vstack(
            (self.scores, ((x - x_hat)**2).sum(axis=1).reshape(-1,1)))                   
        self.projection = torch.vstack((self.projection, z))
        self.labels = torch.vstack((self.labels, y))
        if i == num_batches:
            self.labels = torch.hstack((self.labels, self.scores))
            break

    data_ = data.recover_labels(self.labels[:,:-1])

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
    data.feature_names[13] = 'outlier_score'                               
    for index, feature in data.feature_names.items():
        if index not in [8,9,10]: # exclude day, month, year (isotropic)
            args = dict(size=10,
                        color=self.labels[:,index+1],
                        colorscale=fig.data[0].marker.colorscale 
                        if index != 13 else '',
                        line=dict(width=0.5,
                                  color='DarkSlateGrey')
                       )
              
            #create a button object for the feature we are on
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
            
    Attributes:
        linear1: 
            torch.nn.Linear, input layer of encoder
        linear2: 
            torch.nn.Linear, hidden layer of encoder
        linear3: 
            torch.nn.Linear, input layer of decoder
        linear4: 
            torch.nn.Linear, hidden layer of decoder
    
    Methods:
        encode: 
            forward pass through the encoder network
            
            args: 
                x, torch.utils.data.DataLoader object that
                contains the dataset of interest
        
        decode: 
            forward pass through the decoder network
            
            args: 
                x, encoded result from the self.encode method
    
        fit: 
            backward pass through the network,
            optimizing over parameter space by minimizing
            squared reconstruction error.
            
            args: 
                data: 
                    torch.utils.data.DataLoader object 
                    that contains the dataset of interest
                
                epochs:
                    number of times to batch cycle through 
                    all of the data being learned from. Default:1
        
        plot_loss: 
            generate a plot of the reconstruction loss 
            for each batch observed during training                
        
        plot_projection: 
            generate a plot of the encoded data in
            low-dimensional space.
            
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
        
    def encode(self, x):
        x = F.leaky_relu(
            self.bn1(self.linear1(x)) 
            if self.batch_norm 
            else self.linear1(x))
        z = self.linear2(x)
        return z
    
    def decode(self, z):
        z = F.leaky_relu(
            self.bn2(self.linear3(z)) 
            if self.batch_norm 
            else self.linear3(z))
        x_hat = torch.sigmoid(self.linear4(z))
        return x_hat
    
    def fit(self, data, epochs=1):
        '''
        Train the autoencoder network
        '''
        self.losses = []
        opt = torch.optim.Adam(self.parameters())
        for epoch in range(epochs):
            start_time = time.time()
            for x, y in data:
                x = x.to(device) # use GPU if available
                opt.zero_grad()
                x_hat = self.decode(self.encode(x))
                loss = ((x - x_hat)**2).sum()
                self.losses.append(loss)
                loss.backward()
                opt.step()
            print(f'epoch [{epoch+1}/{epochs}] | ' +
                  f'loss: {round(loss.detach().item(), 4)} | '+
                  f'elapsed time: {round((time.time() - start_time)/60, 2)} minutes')
            
    plot_projection = plot_projection
    plot_loss = plot_loss
    
class VAE(nn.Module):
    '''
    Variational autoencoder class to learn latent representation of the data.
    Uses modified loss function that aims to minimize squared reconstruction error
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
    
    Attributes:
        linear1: 
            torch.nn.Linear, input layer of encoder
        linear2: 
            torch.nn.Linear, hidden layer of encoder, representing mu
        linear3: 
            torch.nn.Linear, input layer of decoder, representing sigma
        linear4: 
            torch.nn.Linear, hidden layer of decoder
    
    Methods:
        encode: 
            forward pass through the encoder network
            
            args: 
                x, torch.utils.data.DataLoader object that
                contains the dataset of interest
        
        decode: 
            forward pass through the decoder network
            
            args: 
            x, encoded result from the self.encode method
    
        fit: 
            backward pass through the network,
            optimizing over parameter space by minimizing
            squared reconstruction error.
            
            args: 
                data, torch.utils.data.DataLoader object 
                that contains the dataset of interest

                epochs, number of times to batch cycle through 
                all of the data being learned from. default=1
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
        self.kl = 0
        
        # can use for sampling on GPU:
        # self.N.loc = self.N.loc.cuda()
        # self.N.scale = self.N.scale.cuda()
        
    def encode(self, x):
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
        z = F.leaky_relu(
            self.bn2(self.linear4(z)) 
            if self.batch_norm 
            else self.linear4(z))
        x_hat = torch.sigmoid(self.linear5(z))
        return x_hat
    
    def fit(self, data, epochs=1, gamma=1):
        self.losses = []
        opt = torch.optim.Adam(self.parameters())
        for epoch in range(epochs):
            start_time = time.time()
            for x, y in data:
                x = x.to(device) # use GPU if available
                opt.zero_grad()
                x_hat = self.decode(self.encode(x))
                loss = ((x - x_hat)**2).sum() + gamma*self.kl
                self.losses.append(loss)
                loss.backward()
                opt.step()
            print(f'epoch [{epoch+1}/{epochs}] | ' +
                  f'loss: {round(loss.detach().item(), 4)} | '+
                  f'elapsed time: {round((time.time() - start_time)/60, 2)} minutes')
                
    plot_projection = plot_projection
    plot_loss = plot_loss
                
def main():
    if len(sys.argv) == 4:
        start_time = time.time()
        philly_payments_clean, model_choice, epochs = sys.argv[1:]
        
        print(f'Loading dataset from: {philly_payments_clean}')
        data = torch.load(philly_payments_clean)
        dims = data.X_.shape[1]
        
        print('Training the autoencoder...')
        model = AE(dims,2,dims) if model_choice == 'ae' else VAE(dims,2,dims)
        model.fit(data.dataset, epochs=int(epochs))
        
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

            Example: python scripts/models.py data/philly_payments_clean ae 5
            '''
        )
            
if __name__ == '__main__':
    main()
