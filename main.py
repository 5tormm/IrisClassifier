import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
from matplotlib import pyplot
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
pyplot.plot()
shell = TerminalInteractiveShell.instance()
shell.get_ipython().run_line_magic('matplotlib', 'inline')
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

#Create Model Class that inherits nn.Module
class Model(nn.Module):
    #Input Layer -> 
    #Hidden Layer(number of neurons) -> 
    #Other hidden layer... -> Output
    def __init__(self, in_features=4, h1=8, h2=9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)
#pick random seed    
torch.manual_seed(69)
#Create instance of model
model = Model()
# Change last column from strings to ints
my_df['species'] = my_df['species'].replace("setosa", 0)
my_df['species'] = my_df['species'].replace("versicolor", 1)
my_df['species'] = my_df['species'].replace("virginica", 2)

# Train Test Split! Set x,y
X = my_df.drop('species', axis=1)
y = my_df['species']

# Convert to numpy array
X = X.values
y = y.values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Convert X feature to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set how far off the prediction are 
criterion = nn.CrossEntropyLoss()
#Choose Adam Optimizer, lr = learning rate(if error doesn't go down, lower it)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train model
#Epochs? one run through all data
epochs = 200
losses = []
for i in range(epochs):
    #GO forwards and get prediciton
    y_pred = model.forward(X_train) #Get predicted results

    #Measure loss/error
    loss = criterion(y_pred, y_train) #predicted vs y_train

    #Keep track of losses
    losses.append(loss.detach().numpy())

    #print every 10 epoch
    #if i % 10 == 0:
        #print(f'Epoch : {i} and loss: {loss}')

    #Back propagation: take error rate of forwards propegation and feed it back thru network to finetune weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Evalutate Model
with torch.no_grad(): #Turn off back propogation
    y_eval = model.forward(X_test) #X_test are features from test set, y_eval = predictions
    loss = criterion(y_eval, y_test) #loss, y_eval vs y_test
correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        #Iris type network thinks is
        #print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t{y_val.argmax().item()}')

        #Correct?
        if y_val.argmax().item() == y_test[i]:
            correct += 1
#print(f'Got {correct} correct')
sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
pedal_length = float(input("Pedal Length: "))
pedal_width = float(input("Pedal Width: "))
new_iris = torch.tensor([sepal_length, sepal_width, pedal_length, pedal_width])
with torch.no_grad():
    val = model(new_iris)
    if int(val.argmax().item()) == 0:
        x = "Setosa"

    if int(val.argmax().item()) == 1:
        x = "Versicolor"

    if int(val.argmax().item()) == 2:
        x = "Virginica"

    print(f'This iris is a {x}')
