import torch
import torch.nn as nn
import torchvision





def test1():
    # setup
    model = nn.Linear(10, 6, bias=False)
    model2 = nn.Linear(10, 6, bias=False)

    x1 = torch.randn(1, 10)
    x2 = torch.randn(1, 10)

    y1 = torch.randn(1, 6)
    y2 = torch.randn(1, 6)

    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.07,1.19,1,1.01,0.81,0.99]))
    criterion2 = torch.nn.CrossEntropyLoss()

    # sum
    model.train()
    with torch.set_grad_enabled(True):
        out1 = model(x1)
        loss = criterion(out1, y1)
        loss.backward()
        out2 = model(x2)
        loss = criterion(out2, y2)
        loss.backward()
    # Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward


def test2():
    model = torchvision.models.resnet18(pretrained=True, )
    #model.fc.in_features = torch.nn.Linear(model.fc.in_features, 6)  # add last layer

    #lin = model.fc
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.fc.in_features, 6),  # last layer
    )
    #model.fc = new_lin

    print(model)
    x = torch.randn(1,3, 244,244)
    y = torch.randn(1, 6)

    out = model(x)


    return 1



if __name__ == "__main__":
    x = test2()
