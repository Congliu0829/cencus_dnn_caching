from utils import *
from basicnet import *

class LDSharedModel(object):

 def __init__(self, params):
    self.device = params.get('device', 'cuda')
    for key, val in params.items():
        setattr(self, key, val)
    self.input_size = self.X_val.shape[1]
    train_tensor = TensorDataset(Tensor(self.X_train), Tensor(self.y_train))
    self.train_loader = DataLoader(dataset = train_tensor, batch_size = self.bs, shuffle = True)
    self.fitted_model =  None

 def predict(self, X_test):
      """
      Return Prediction for Future Test Data
      """
      # if 'numpy' not in str(type(X_test)):
      #   X_test= X_test.values
      X_test = torch.FloatTensor(X_test).to(self.device)
      all_pred = []
      for i in range(len(X_test)):
        all_pred.append(self.fitted_model.forward(X_test[i].view(1,-1)).detach().numpy())
      pred = np.concatenate(all_pred)

      return  (pred>=0.5).astype(int), pred

 def _model_eval(self, model):
     """
     Evaluate current model based on Accuracy and Fairness Score (DI-score)
     """
     self.fitted_model = copy.deepcopy(model)
     y_pred_val,_ = self.predict(self.X_val)
     acc = compute_fairness_score(self.y_val, y_pred_val)

     return acc

 def fit(self, options):

     #train_verbose = options.get('train_verbose', False)
     seed = options.get('seed', 0)
     model_lr = options.get('model_lr', 1e-2)
     epochs = options.get('epochs', 200)
     step_size = options['step_size']

     lr_mult = options['lr_mult'] # Initial Lagrangian multiplier, 0 for LD method
     return_output = options.get('return_output', False)
     torch.manual_seed(seed)
     model =  Net(12, [10,5])
     bce_criterion = nn.BCELoss(reduce='mean')
     optimizer = torch.optim.Adam(model.parameters(), lr = model_lr)
     logs = []

     for epoch in range(epochs):
         violation_list = []
         for input_train, target_train in self.train_loader:
             model.train()
             input_train = input_train.to(self.device)
             target_train = target_train.to(self.device)
             loss = torch.tensor(0.0)
             mean_output_list = []
             optimizer.zero_grad()
             output = model.forward(input_train)
             loss += bce_criterion(output, target_train)
             mean_output_list.append(torch.mean(output))
             if len(mean_output_list) > 1:
                 # add the mean difference between two groups
                 violation = torch.abs(mean_output_list[0] - mean_output_list[1])
                 loss += lr_mult * violation
                 violation_list.append(violation.item())

             loss.backward()
             optimizer.step()

         acc = self._model_eval(model)
         logs.append(copy.deepcopy([acc, lr_mult]))

         lr_mult += step_size * np.mean(violation_list)

     self.fitted_model = model
     self.best_options = options
     self.logs = logs
     self.best_acc = acc

     if return_output:
        return model, acc, logs

 def hyper_opt(self, grid_search_list):

     best_metric = -np.inf
     best_model = None
     best_options = None
     best_logs = None
     for options in grid_search_list:
         print( options)
         options['return_output']  = True
         curr_model, curr_acc, logs = self.fit(options)
         if options['acc_only'] :
             curr_metric = curr_acc
         else:
             curr_metric = logs[-1][0] - logs[-1][1] # Acc - DI-score as a heuristic rule to choose the model

         if curr_metric > best_metric:
             best_metric = curr_metric
             best_model = curr_model
             best_options = options
             best_logs = logs

     self.model = best_model
     self.best_metric = best_metric
     self.best_options = best_options
     self.logs = best_logs

 def get_embedding(self, x, k):
    assert k in list(range(1, 3))
    for i in range(2*k): # include activation
        x = self.model._layers[i](x)
    return x

















