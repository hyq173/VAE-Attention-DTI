import pickle
from utils import *
from DL_ClassifierModel import *
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())


def bindingDB():
    # dataPath = 'E:\\code\\python\\bridge-dpi\\data\\bindingDB\\'
    dataPath = './data/bindingDB'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    savePath = './save_pkl/pkl-bindingdb'

    mkdir('./save_pkl')
    # mkdir('./dataClass')
    # dataClass = DataClass(dataPath= dataPath, pSeqMaxLen=1024, dSeqMaxLen=128)
    # pickle.dump(dataClass, file=open('./dataClass/dataClass_bindingDB.pkl', 'wb'))
    dataClass = pickle.load(file=open('./dataClass/dataClass_bindingDB.pkl','rb'))

    # # train the models
    model = DTI_Bridge(outSize=128,
                    cHiddenSizeList=[1024],
                    fHiddenSizeList=[1024,256],
                    fSize=1024, cSize=dataClass.pContFeat.shape[1],
                    gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                    hdnDropout=0.5, fcDropout=0.5, device=device)
    print('===========begin to train===========')
    model.train(dataClass, log = savePath+'_log.txt', trainSize=128, batchSize=128, epoch=128,
                lr=0.001, stopRounds=-1, earlyStop=30,
                savePath=savePath, metrics="ACC", report=["ACC", "AUC", "Precision", "Recall", "F1"],
                preheat=0)

def celegans():
    # dataPath = 'E:\\code\\python\\bridge-dpi\\data\\bindingDB\\'
    dataPath = './data/celegans/original/data.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    savePath = './save_pkl/pkl-celegans'

    # mkdir('./save_pkl')
    # mkdir('./dataClass')
    # dataClass = DataClass_normal(dataPath=dataPath, pSeqMaxLen=1024, dSeqMaxLen=128, sep = ' ')
    # pickle.dump(dataClass, file=open('./dataClass/dataClass_celegans.pkl', 'wb'))
    dataClass = pickle.load(file=open('./dataClass/dataClass_celegans.pkl','rb'))

    # # train the models
    model = DTI_Bridge(outSize=128,
                    cHiddenSizeList=[1024],
                    fHiddenSizeList=[1024,256],
                    fSize=1024, cSize=dataClass.pContFeat.shape[1],
                    gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                    hdnDropout=0.5, fcDropout=0.5, device=device)
    print('===========begin to train===========')
    model.train(dataClass, log = savePath+'_log.txt', trainSize=128, batchSize=128, epoch=128,
                lr=0.001, stopRounds=-1, earlyStop=30, savePath=savePath, 
                metrics="ACC", report=["ACC", "AUC", "Precision", "Recall", "F1"], preheat=0)

def human():
    # dataPath = 'E:\\code\\python\\bridge-dpi\\data\\bindingDB\\'
    dataPath = './data/human/original/data.txt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    savePath = './save_pkl/pkl-human'

    # mkdir('./save_pkl')
    # mkdir('./dataClass')
    # dataClass = DataClass_normal(dataPath= dataPath, pSeqMaxLen=1024, dSeqMaxLen=128, sep = ' ')
    # pickle.dump(dataClass, file=open('./dataClass/dataClass_human.pkl', 'wb'))
    dataClass = pickle.load(file=open('./dataClass/dataClass_human.pkl','rb'))

    # # train the models
    model = DTI_Bridge(outSize=128,
                    cHiddenSizeList=[1024],
                    fHiddenSizeList=[1024,256],
                    fSize=1024, cSize=dataClass.pContFeat.shape[1],
                    gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                    hdnDropout=0.5, fcDropout=0.5, device=device)
    print('===========begin to train===========')
    model.train(dataClass, log = savePath+'_log.txt', trainSize=128, batchSize=128, epoch=128,
                lr=0.001, stopRounds=-1, earlyStop=30, savePath=savePath, 
                metrics="ACC", report=["ACC", "AUC", "Precision", "Recall", "F1"], preheat=0)

# bindingDB()
# celegans()
human()