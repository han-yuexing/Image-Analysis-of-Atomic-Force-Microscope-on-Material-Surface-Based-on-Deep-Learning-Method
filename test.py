from PIL import Image
import numpy as np
from net import VBN
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from Data import Tai
from torch.utils import data
import tqdm

def test():
    model_path = './model_best.pth.tar'
    TestSet = Tai(X_files = "./patch", test=True)
    test_loader = data.DataLoader(TestSet, batch_size=4096)
    
    model = VBN()
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    # torch.save(torch.load(model_path)['model_state_dict'], './model_best.pth.tar')
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    Pre = np.array([])
    for batch_idx, image in tqdm.tqdm(
                enumerate(test_loader), total=len(test_loader)):
        if torch.cuda.is_available():
            image = image.cuda()
        image = Variable(image)
        with torch.no_grad():
            score = model(image)
        # print(score)
        lbl_pred = score.max(1)[1].cpu().numpy()
        # print(lbl_pred)
        # print((lbl_pred.flatten()).shape)
        Pre = np.hstack((Pre, lbl_pred.flatten()))
    np.save("./pre.npy", Pre)
    
        


    

if __name__ == "__main__":
    # test()
    pre = np.load("./pre.npy")
    lbl = np.load("./Label_12.npy")
    print(pre.shape, set(pre))
    print(lbl)
    print(lbl.shape, set(lbl))
    print(accuracy_score(np.load("./pre.npy"), np.load("./Label_12.npy")))
