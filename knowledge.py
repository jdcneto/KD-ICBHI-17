import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from callback import EarlyStopping
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix



class KD():
  def __init__(self, 
               teacher, 
               student, 
               train_loader, 
               val_loader,
               opt_student,
               temp,
               alpha,
               dir='trained/'
               
               ):
    
    self.device = "cuda" if torch.cuda.is_available else "cpu"
    
    self.teacher = teacher
    self.student = student
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.opt_student = opt_student
    self.temp = temp
    self.alpha = alpha
    self.ce_fn = nn.CrossEntropyLoss()
    
    self.teacher = self.teacher.to(self.device)
    self.student = self.student.to(self.device)
    self.dir = dir
    
    if not os.path.exists(self.dir):
        os.makedirs(self.dir)
  
  
  def KD_loss(self, y_pred_student, y_pred_teacher, y_true):
    """Soft teacher/student cross entropy loss from [Hinton et al (2015)]
        (https://arxiv.org/abs/1503.02531)
    """
    # hard loss
    hard_loss = F.cross_entropy(y_pred_student, y_true)

    # soft loss
    soft_teacher = F.softmax(y_pred_teacher / self.temp, dim=-1)
    soft_student = F.log_softmax(y_pred_student / self.temp, dim=-1)
    soft_loss = -(self.temp ** 2 * soft_teacher * soft_student).sum(-1).mean()

    loss = self.alpha*hard_loss + (1-self.alpha)*soft_loss
    return loss
    
  
  def KL_loss(self, y_pred_student, y_pred_teacher, y_true):
    """Soft teacher/student cross entropy loss from [Hinton et al (2015)]
        (https://arxiv.org/abs/1503.02531)
    """
    # hard loss
    hard_loss = F.cross_entropy(y_pred_student, y_true)
    
    # soft loss
    soft_teacher = F.softmax(y_pred_teacher / self.temp, dim=1)
    soft_student = F.log_softmax(y_pred_student / self.temp, dim=1)
    
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    
    loss = self.alpha*hard_loss + (1-self.alpha)*(self.temp**2)*soft_loss

    return loss
    
    
  def CMKD_loss(self, y_pred_student, y_pred_teacher, y_true):
    """Soft teacher/student cross entropy loss from [Gong et al (2022)]
        (https://arxiv.org/abs/2203.06760)
    """
    # hard loss
    hard_loss = F.cross_entropy(y_pred_student, y_true)
    
    # soft loss
    soft_teacher = F.softmax(y_pred_teacher / self.temp, dim=1)
    soft_student = F.log_softmax(y_pred_student, dim=1)
    
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    
    loss = self.alpha*hard_loss + (1-self.alpha)*soft_loss

    return loss
    
  
  def fit_teacher(self, epochs=100, patience=15, verbose=True):
    loss_hist_train = [0] * epochs
    accuracy_hist_train = [0] * epochs
    loss_hist_valid = [0] * epochs
    accuracy_hist_valid = [0] * epochs
    y_pred = []
    y_true = []
    teacher_path = os.path.join(self.dir, 'teacher.pt')

    # initialize the early stopping 
    early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=teacher_path)

    for epoch in range(epochs):
        # Train 
        self.student.train()
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device) 
            y_batch = y_batch.to(self.device) 
            # Prediction
            pred_t = self.teacher(x_batch)
            # Loss 
            loss = self.ce_fn(pred_t, y_batch)
            loss.backward()
            self.opt_student.step()
            self.opt_student.zero_grad()

            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred_t, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(self.train_loader.dataset)
        accuracy_hist_train[epoch] /= len(self.train_loader.dataset)
        
        # Validation
        self.teacher.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.to(self.device) 
                pred = self.teacher(x_batch)
                loss = self.ce_fn(pred, y_batch)

                loss_hist_valid[epoch] += loss.item()*y_batch.size(0) 
                output = torch.argmax(pred, dim=1, keepdim=False)
                is_correct = (output == y_batch).float() 
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()
                
                # Storing 
                # if batch size len == 1
                if len(pred)==1:
                    y_pred.append(output.item())
                    y_true.append(y_batch.item())
                else: 
                    y_pred.extend(output.tolist())
                    y_true.extend(y_batch.tolist())
                    
                cnf_matrix= confusion_matrix(y_true, y_pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        # Average Score 
        Score = (TPR.mean()+TNR.mean())/2

        loss_hist_valid[epoch] /= len(self.val_loader.dataset)
        accuracy_hist_valid[epoch] /= len(self.val_loader.dataset)
       
        print(f'Epoch {epoch+1} Train accuracy: {accuracy_hist_train[epoch]:.4f} Val accuracy: {accuracy_hist_valid[epoch]:.4f} Val loss: {loss_hist_valid[epoch]:.4f} Score: {Score:.4f} ')
    
        # early_stopping needs the validation score to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(Score, self.teacher)
        
        if early_stopping.early_stop:
            print("Stopping Teacher Training")
            break
    
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid
  
  
  def fit_student(self, epochs=100, patience=15, verbose=True):
    loss_hist_train = [0] * epochs
    accuracy_hist_train = [0] * epochs
    loss_hist_valid = [0] * epochs
    accuracy_hist_valid = [0] * epochs
    y_pred = []
    y_true = []
    self.teacher.load.state_dict(self.teacher_dir)
    # initialize the early stopping 
    self.student_path = os.path.join(self.dir, 'student.pt')
    
    early_stopping = EarlyStopping(patience=patience, verbose=verbose, path=self.student_path)

    for epoch in range(epochs):
        # Train 
        self.student.train()
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device) 
            y_batch = y_batch.to(self.device) 
            # Prediction
            pred_s = self.student(x_batch)
            pred_t = self.teacher(x_batch)
            # Loss 
            #loss = self.KL_loss(pred_s, pred_t, y_batch)
            #loss = self.KD_loss(pred_s, pred_t, y_batch)
            loss = self.CMKD_loss(pred_s, pred_t, y_batch)
            loss.backward()
            self.opt_student.step()
            self.opt_student.zero_grad()

            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred_s, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(self.train_loader.dataset)
        accuracy_hist_train[epoch] /= len(self.train_loader.dataset)
        
        # Validation
        self.student.eval()
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device) 
                y_batch = y_batch.to(self.device) 
                pred = self.student(x_batch)
                loss = self.ce_fn(pred, y_batch)

                loss_hist_valid[epoch] += loss.item()*y_batch.size(0) 
                output = torch.argmax(pred, dim=1, keepdim=False)
                is_correct = (output == y_batch).float() 
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()
                
                # Storing 
                # if batch size len == 1
                if len(pred)==1:
                    y_pred.append(output.item())
                    y_true.append(y_batch.item())
                else: 
                    y_pred.extend(output.tolist())
                    y_true.extend(y_batch.tolist())
                    
                cnf_matrix= confusion_matrix(y_true, y_pred)

        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        # Average Score 
        Score = (TPR.mean()+TNR.mean())/2

        loss_hist_valid[epoch] /= len(self.val_loader.dataset)
        accuracy_hist_valid[epoch] /= len(self.val_loader.dataset)
        
        print(f'Epoch {epoch+1} Train accuracy: {accuracy_hist_train[epoch]:.4f} Val accuracy: {accuracy_hist_valid[epoch]:.4f} Val loss: {loss_hist_valid[epoch]:.4f} Score: {Score:.4f} ')
    
        # early_stopping needs the validation score to check if it has increased, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(Score, self.student)
        
        if early_stopping.early_stop:
            print("Stopping Student Training")
            break
    
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


  def evaluate(self, test_loader):
    """
    Evaluate the student model's accuracy at test set.
    """
    state_dict = torch.load(self.student_path)
    model = self.student.load_state_dict(state_dict) 
    
    model.eval();
    length_of_dataset = len(test_loader.dataset)
    correct = 0
    test_loss = 0
    
    score_list   = []
    pred_list    = []
    target_list  = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            
            loss = self.ce_fn(output, target)
            test_loss +=loss*target.size(0)
            
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target).sum().item()
            
            # Storing 
            if len(pred)==1:
                pred_list.append(pred.item())
                target_list.append(target.item())
            else:
                pred_list.extend(pred.tolist())
                target_list.extend(target.tolist())
            score_list.extend(nn.Softmax(dim = 1)(output).tolist())
              
    Accuracy = correct / length_of_dataset       
          
    # metrics
    F1_score = f1_score(target_list, pred_list, average="macro") 
    Accuracy = accuracy_score(target_list, pred_list) 
    Roc_AUC  = roc_auc_score(target_list, score_list, average="macro",multi_class='ovo')
    cnf_matrix= confusion_matrix(target_list, pred_list)
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Average Score 
    Score = (TPR.mean()+TNR.mean())/2
    
    # put values into dictionary
    metrics_dict = {"Accuracy": Accuracy,
                  "F1-score": F1_score,
                  "Roc_AUC":Roc_AUC,
                  "Loss": test_loss/length_of_dataset,
                  "CM":cnf_matrix,
                  "Target":target_list,
                  "Predict":pred_list,
                  "Sensitivity":TPR.mean(),
                  "Specificity":TNR.mean(),
                  "Score": Score}
    
   
    return metrics_dict