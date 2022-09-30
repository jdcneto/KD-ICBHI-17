import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix


class BaseClass:
    """
    Basic implementation of a general Knowledge Distillation framework

    :param teacher_model (torch.nn.Module): Teacher model
    :param student_model (torch.nn.Module): Student model
    :param train_loader (torch.utils.data.DataLoader): Dataloader for training
    :param val_loader (torch.utils.data.DataLoader): Dataloader for validation/testing
    :param optimizer_teacher (torch.optim.*): Optimizer used for training teacher
    :param optimizer_student (torch.optim.*): Optimizer used for training student
    :param loss_fn (torch.nn.Module): Loss Function used for distillation
    :param temp (float): Temperature parameter for distillation
    :param distil_weight (float): Weight paramter for distillation loss
    :param device (str): Device used for training; 'cpu' for cpu and 'cuda' for gpu
    :param log (bool): True if logging required
    :param logdir (str): Directory for storing logs
    """

    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        test_loader,
        optimizer_teacher,
        optimizer_student,
        loss_fn=nn.KLDivLoss(),
        temp=20.0,
        distil_weight=0.5,
        log=False,
        logdir="./Experiments",
    ):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer_teacher = optimizer_teacher
        self.optimizer_student = optimizer_student
        self.temp = temp
        self.distil_weight = distil_weight
        self.log = log
        self.logdir = logdir

        if self.log:
            self.writer = SummaryWriter(logdir)

        self.device = "cuda:0" if torch.cuda.is_available else "cpu"
        
        if teacher_model:
            self.teacher_model = teacher_model.to(self.device)
        else:
            print("Warning!!! Teacher is NONE.")

        self.student_model = student_model.to(self.device)
        self.loss_fn = loss_fn
        self.ce_fn = nn.CrossEntropyLoss()
    
    ### Train Teacher
    def train_teacher(
        self,
        epochs=20,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/teacher.pt",
    ):
        """
        Function that will be training the teacher

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the teacher model
        :param save_model_pth (str): Path where you want to store the teacher model
        """
        self.teacher_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_teacher_model_weights = deepcopy(self.teacher_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print("Training Teacher... ")
        
        scheduler = StepLR(self.optimizer_teacher, step_size=50, gamma=0.5)

        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0
            for (data, label) in self.train_loader:
                data = data.to(self.device)
                label = label.type(torch.LongTensor)
                label = label.to(self.device)
                out = self.teacher_model(data)

                pred = out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                loss = self.ce_fn(out, label)

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                epoch_loss += loss.item()*label.size(0)

            epoch_acc = correct / length_of_dataset
            epoch_loss = epoch_loss/length_of_dataset

            epoch_val_loss, epoch_val_acc = self._evaluate_model(self.teacher_model)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_teacher_model_weights = deepcopy(
                    self.teacher_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Teacher", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Teacher", epoch_acc, epochs)
                self.writer.add_scalar("Validation loss/Teacher", epoch_val_loss, epochs)
                self.writer.add_scalar("Validation accuracy/Teacher", epoch_val_acc, epochs)

            loss_arr.append(epoch_loss)
            print("Teacher Stats --> Epoch: {} | Train Loss: {:.2f}, Train Accuracy: {:.2f} | Validation Loss: {:.2f}, Validation Accuracy: {:.2f}".format(ep + 1, epoch_loss, 
            epoch_acc, epoch_val_loss, epoch_val_accuracy))

            self.post_epoch_call(ep)
            scheduler.step()

        self.teacher_model.load_state_dict(self.best_teacher_model_weights)
        if save_model:
            torch.save(self.teacher_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)
    
    ### Train Student
    def _train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pt",
    ):
        """
        Function to train student model - for internal use only.

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self.teacher_model.eval()
        self.student_model.train()
        loss_arr = []
        length_of_dataset = len(self.train_loader.dataset)
        best_acc = 0.0
        self.best_student_model_weights = deepcopy(self.student_model.state_dict())

        save_dir = os.path.dirname(save_model_pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Training Student...")

        scheduler = StepLR(self.optimizer_student, step=50, gamma=0.5)
        for ep in range(epochs):
            epoch_loss = 0.0
            correct = 0

            for (data, label) in self.train_loader:

                data = data.to(self.device)
                label = label.type(torch.LongTensor)
                label = label.to(self.device)

                student_out = self.student_model(data)
                teacher_out = self.teacher_model(data)

                loss = self.calculate_kd_loss(student_out, teacher_out, label)

                pred = student_out.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                epoch_loss += loss.item()*label.size(0)

            epoch_acc = correct / length_of_dataset
            epoch_loss = epoch_loss/length_of_dataset

            epoch_val_loss, epoch_val_acc = self._evaluate_model(self.student_model)

            if epoch_val_acc > best_acc:
                best_acc = epoch_val_acc
                self.best_student_model_weights = deepcopy(
                    self.student_model.state_dict()
                )

            if self.log:
                self.writer.add_scalar("Training loss/Student", epoch_loss, epochs)
                self.writer.add_scalar("Training accuracy/Student", epoch_acc, epochs)
                self.writer.add_scalar("Validation accuracy/Student", epoch_val_acc, epochs)
                self.writer.add_scalar("Validation loss/Student", epoch_val_loss, epochs)

            loss_arr.append(epoch_loss)
            print("Student Stats --> Epoch: {} | Train Loss: {:.2f}, Train Accuracy: {:.2f} | Validation Loss: {:.2f}, Validation Accuracy: {:.2f}".format(ep + 1, epoch_loss, 
            epoch_acc, epoch_val_loss, epoch_val_accuracy))
            
            scheduler.step()

        self.student_model.load_state_dict(self.best_student_model_weights)
        if save_model:
            torch.save(self.student_model.state_dict(), save_model_pth)
        if plot_losses:
            plt.plot(loss_arr)

    def train_student(
        self,
        epochs=10,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pt",
    ):
        """
        Function that will be training the student

        :param epochs (int): Number of epochs you want to train the teacher
        :param plot_losses (bool): True if you want to plot the losses
        :param save_model (bool): True if you want to save the student model
        :param save_model_pth (str): Path where you want to save the student model
        """
        self._train_student(epochs, plot_losses, save_model, save_model_pth)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Custom loss function to calculate the KD loss for various implementations

        :param y_pred_student (Tensor): Predicted outputs from the student network
        :param y_pred_teacher (Tensor): Predicted outputs from the teacher network
        :param y_true (Tensor): True labels
        """

        raise NotImplementedError

    def _evaluate_model(self, model, verbose=False):
        """
        Evaluate the given model's accuracy over val set.
        For internal use only.

        :param model (nn.Module): Model to be used for evaluation
        :param verbose (bool): Display Accuracy
        """
        model.eval()
        length_of_dataset = len(self.val_loader.dataset)
        epoch_loss = 0
        correct = 0
        outputs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.type(torch.LongTensor)
                target = target.to(self.device)
                output = model(data)
                
                loss = self.ce_fn(output, target)
                epoch_loss +=loss*target.size(0)

                outputs.append(output)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / length_of_dataset
        loss = epoch_loss/length_of_dataset

        if verbose:
            print("-" * 80)
            print("Validation Accuracy: {:.2%} | Validation Loss {:.2f}".format(accuracy, loss))
        return loss, accuracy

    def evaluate(self, teacher=False):
        """
        Evaluate method for printing accuracies of the trained network

        :param teacher (bool): True if you want accuracy of the teacher network
        """
        if teacher:
            model = deepcopy(self.teacher_model).to(self.device)
        else:
            model = deepcopy(self.student_model).to(self.device)
        loss, accuracy = self._evaluate_model(model)

        return accuracy, loss
        
    def inference(self):
        """
        Evaluate the student model's accuracy over test set.
        For internal use only.
       
        """
        model = self.student_model()
        model.load_state_dict(torch.load("./models/student.pt", map_location=self.device))
        model.eval()
        length_of_dataset = len(self.test_loader.dataset)
        correct = 0
        test_loss = 0
        outputs = []
        
        score_list   = []
        pred_list    = []
        target_list  = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.type(torch.LongTensor)
                target = target.to(self.device)
                output = model(data)
                
                loss = self.ce_fn(output, target)
                test_loss +=loss*target.size(0)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Storing 
                if len(pred)==1:
                    pred_list.append(pred.squeeze().tolist())
                    target_list.append(target.squeeze().tolist())
                else:    
                    pred_list.extend(pred.squeeze().tolist())
                    target_list.extend(target.squeeze().tolist())
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
        
        print('Inference Finished in') 
        return metrics_dict     
       
               
    def get_parameters(self):
        """
        Get the number of parameters for the teacher and the student network
        """
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())

        print("-" * 80)
        print("Total parameters for the teacher network are: {}".format(teacher_params))
        print("Total parameters for the student network are: {}".format(student_params))

    def post_epoch_call(self, epoch):
        """
        Any changes to be made after an epoch is completed.

        :param epoch (int) : current epoch number
        :return            : nothing (void)
        """

        pass