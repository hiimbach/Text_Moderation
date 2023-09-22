import datetime
import torch 
import os 
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from sentence_transformers import SentenceTransformer

from utils.data import CustomDataset, data_split


def accuracy(correct, total):
    return torch.sum(torch.tensor(correct)).item() /len(total)


def compare(outputs, labels):
    _, out_pred = torch.max(outputs, dim = 1)
    _, label_pred = torch.max(labels, dim=1)
    return torch.sum(out_pred == label_pred)


class TrainingLoop():
    '''
    Create train task to train nn.Module model
    
    '''
    def __init__(self, model: str, 
                        csv_data_path: str, 
                        batch_size: int, 
                        loss_fn, 
                        optim_fn: torch.optim, 
                        lr: float,  
                        data_split_ratio = 0.8,
                        device='cpu'):
        
        self.layer = nn.Linear(768, 5)
        self.model = SentenceTransformer(model).to(device)
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optim_fn(self.layer.parameters(), lr)
        
        # Prepare data for training and evaluation
        train_idx, val_idx = data_split(csv_data_path, split_ratio=data_split_ratio)
        train_dataset = CustomDataset(csv_data_path, train_idx)
        val_dataset = CustomDataset(csv_data_path, val_idx)
        
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        self.val_total = len(val_dataset)
        
        # Define training device 
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.loss_fn.to(self.device)
        self.layer.to(self.device)
        
    def train(self, n_epochs, save_name, eval_interval=5, pretrained_weight=None):
        '''
        
        '''
        
        # Load pretrained weight
        if pretrained_weight:
            self.layer.load_state_dict(torch.load(pretrained_weight, map_location=self.device))
        
        # Prepare for saving 
        save_path = os.path.join('runs', f"{save_name}")
        if os.path.exists(save_path):
            i = 1
            while os.path.exists(f"{save_path}_{i}"):
                i += 1
            save_path = f"{save_path}_{i}"
            
        save_weight_path = os.path.join(save_path, "weights")
        os.makedirs(save_path)
        os.mkdir(save_weight_path)
        
        # Save tensor log
        writer = SummaryWriter(save_path)
        
        # Max accuracy - used to find best checkpoint
        max_acc = 0
        
        # Train loop
        print(f"{datetime.datetime.now()} Start train on device {self.device}")
        self.layer.train()
        
        for epoch in range(1, n_epochs + 1):  
            
            # Training Phase
            print(f"Epoch {epoch}")
            train_losses = []
            
            # Batch training
            for text, labels in tqdm(self.train_loader, desc="Training"):
                # Write images to tensorboard
                try:
                    writer.add_text('Training_text', ' || '.join(str(text)))
                except:
                    print(text)
                # Predict
                embeds = self.model.encode(text)
                out = self.layer(torch.tensor(embeds))
                
                # import ipdb; ipdb.set_trace()
                train_loss = self.loss_fn(out, labels)
                
                # Backpropagation
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.item())
                
            # End training phase
            mean_train_loss = sum(train_losses)/len(train_losses)
            print(f"{datetime.datetime.now()} Epoch {epoch}: Training loss: {mean_train_loss}")
            
            # Save last checkpoint and write result to tensorboard
            torch.save(self.layer.state_dict(), os.path.join(save_weight_path, "last_ckpt.pt"))
            writer.add_scalar("Loss/train", mean_train_loss, epoch)
            
            ###########################################################################################
            self.layer.eval()
            if epoch == 1 or epoch % eval_interval == 0:
                with torch.no_grad():
                    val_losses = []
                    total_correct = 0
                    total = self.val_total
                    
                    # Batch training
                    for text, labels in tqdm(self.val_loader, desc="Validating"):
                        # Write images to tensorboard
                        try:
                            writer.add_text('Training_text', ' || '.join(text))
                        except:
                            print(text)
                        
                        # Predict
                        embeds = self.model.encode(text)
                        out = self.layer(torch.tensor(embeds))

                        val_loss = self.loss_fn(out, labels)
                        correct = compare(out, labels)  
                        
                        val_losses.append(val_loss.item())    
                        total_correct += correct

                # Calculate loss and accuracy
                acc = total_correct / total
                mean_val_loss = sum(val_losses)/len(val_losses)
                
                # Replace best checkpoint if loss < min_loss:
                if acc > max_acc:
                    max_acc = acc
                    torch.save(self.layer.state_dict(), os.path.join(save_weight_path, "best_ckpt.pt"))
            
                # Write to tensorboard
                writer.add_scalar("Loss/val", mean_val_loss, epoch)
                writer.add_scalar("Accuracy/val", acc, epoch)
                
                # End validating
                print(f"{datetime.datetime.now()} Val Loss {mean_val_loss}")
                print(total_correct, total)
                print(f"{datetime.datetime.now()} Val Accuracy {acc}")
                print("="*70)
                print("")
                
        writer.close()
        return
        