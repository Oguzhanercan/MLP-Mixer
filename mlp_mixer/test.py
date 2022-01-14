import torch
from tqdm import tqdm
import torch.nn as nn

def test(args, testloader, model, criterion):
    model.eval()
    testloss = 0
    correct = 0
    i = -1
    with torch.no_grad():
        for data, target in tqdm(testloader):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            loss = criterion(output, target)
            testloss += loss.item()
            #_, predicted = torch.max(output, 1)
            _,predicted = torch.max(nn.Softmax()(output),dim = 1)
            correct += (predicted == target).sum().item()
            i += 1

    return correct/(i+1)*args.batch_size
