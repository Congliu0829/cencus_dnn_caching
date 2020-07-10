from torch_utils import *
from dataset import Dataset
from basicnet import Net
from torch_utils import *

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

params = {'benchmark': 'adult', 'maxlen': 0, 'batch_size': 10,
          'split': [0.7, 0.2, 0.1], 'seed': 2,
          'epochs': 1, 'lr': 0.001}
data = Dataset(params, 'train', device)

model = Net(data.nfeatures, [10, 4]).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

for epoch in range(params['epochs']):
    for (x, y) in data:
        (x, y) = x.to(device), y.to(device)
        optimizer.zero_grad()
        ŷ = model(x)

        loss = loss_fn(ŷ, y)
        loss.backward()
        optimizer.step()
    print(loss.item())
