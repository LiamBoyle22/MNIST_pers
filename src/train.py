import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import logging 

logging.basicConfig(level=logging.INFO)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

epochs = config["epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
hidden_dim = config["hidden_dim"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(), #converts PIL image to tensor and scales pixel values to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)) #normalizes the tensor with mean and std
])

train_dataset = datasets.MNIST(root="./data", train= True, download= True, transform= transform)
val_dataset = datasets.MNIST(root="./data", train= False, download= True, transform= transform)

train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True) #shuffle to ensure that the data is mixed during training
val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle= False) #no need to shuffle validation data

class MLP(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden_dim) #input layer to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) #hidden layer to hidden layer
        self.fc3 = nn.Linear(hidden_dim, 10) #hidden layer to output layer
        self.relu = nn.ReLU() #activation function
        self.drop = nn.Dropout(0.1) #dropout layer to prevent overfitting

    def forward(self, x):
        x = x.view(x.size(0), -1) #flatten the input image
        x = self.drop(self.relu(self.fc1(x))) #apply first fully connected layer, activation function and dropout
        x = self.drop(self.relu(self.fc2(x))) #apply second fully connected layer, activation function and dropout
        logits = self.fc3(x) #apply output layer
        return logits
    
model = MLP().to(device)

criterion = nn.CrossEntropyLoss() #loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #Adam optimizer with learning rate from config

#quick metric function to calculate accuracy of the model on a batch of data
@torch.no_grad() #disable gradient calculation for validation
def batch_accuracy(logits, y):
    preds = torch.argmax(logits, dim=1) #get predicted class by taking the index of the max logit
    return (preds == y).float().mean().item() #calculate accuracy by comparing predictions with true labels

#validation Loop
@torch.no_grad() 
def validate(model, loader):
    model.eval() #set model to evaluation mode
    total_loss = 0.0 
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device) #move data to device
        logits = model(x) #get model predictions
        loss = criterion(logits, y) #calculate loss

        total_loss += loss.item()
        total_acc += batch_accuracy(logits, y)
        n += 1

    return total_loss / n, total_acc / n #return average loss and accuracy

#training loop
def train_one_epoch(model, loader):
    model.train() #set model to training mode
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for step, (x, y) in enumerate(loader, start= 1):
        x, y = x.to(device) , y.to(device) #move data to device

        logits = model(x) #get model predictions
        loss = criterion(logits, y) #calculate loss

        optimizer.zero_grad(set_to_none= True) #clear gradients
        loss.backward() #backpropagate the loss
        optimizer.step() #update model parameters

        total_loss += loss.item()
        total_acc += batch_accuracy(logits, y) 
        n += 1

        if step % 100 == 0: #print training progress every 100 steps
            print(f"Step {step}, Loss: {total_loss / n:.4f}")

    return total_loss / n, total_acc / n #return average loss and accuracy

epochs = config["epochs"]

for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader) #train the model for one epoch
    val_loss, val_acc = validate(model, val_loader) #validate the model on the validation set

    print(
        f"Epoch {epoch} | "
        f"train loss {tr_loss:.4f},acc {tr_acc:.4f} | "
        f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
    )

torch.save(model.state_dict(), "mnist_mlp_state.pt") #save the model state dictionary to a file
print ("Saved model to mnist_mlp_state.pt")

model2 = MLP().to(device) #create a new instance of the model
model2.load_state_dict(torch.load("mnist_mlp_state.pt")) #load the saved model state dictionary into the new model instance
model2.eval() #set the new model to evaluation mode
print("Loaded!")

#loggging and saving metrics to csv file