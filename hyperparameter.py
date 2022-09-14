import os
class HyperParameter():
    def __init__(self):
        self.num_windows = 32
        self.smi_window_lengths = 5
        self.seq_window_lengths = 7
        self.lamda = -3
        self.num_epoch = 100*3 # 多分类1000
        self.batch_size = 128*2
        self.max_smi_len = 100
        self.max_seq_len = 1000
        self.charsmiset_size = 64
        self.charseqset_size = 25
        self.earlyStop = 30 # 30次acc找不到最大值停止程序 50针对多分类
        self.lr = 0.001*0.1 # *0.1 针对多分类
        self.classify = 5
        self.davis_labels = [0, 1, 2, 3, 4]  # [0, 1, 2, 3, 4, 5, 6]
        self.sub = 'bindingDB'
        self.dataset_path = 'data/'+self.sub
        self.checkpoint_path = 'pth/'+self.sub+'.pth'
        self.log_path = 'log/'+self.sub+'-log.txt'
        self.visual_log = 'visual/'+self.sub
        self.save_affinities = 'result/'+self.sub+'-affinities.txt'
        self.save_preaffinities = 'result/'+self.sub+'-preaffinities.txt'
        self.pkl_path = 'pkl/'+self.sub+'_dataset.pkl'
        self.train_loss_path = 'loss/'+self.sub+'-epoch_loss.txt'
        self.train_acc_path = 'loss/'+self.sub+'-epoch_acc.txt'
        self.train_auc_path = 'loss/'+self.sub+'-epoch_auc.txt'

    def logging(self,msg):
        self.create_dir()
        with open(self.log_path, "a") as fw:
            fw.write("%s\n" % msg)

    def create_dir(self):
        if not os.path.exists('log'):
            os.makedirs('log')
        if not os.path.exists('pth'):
            os.makedirs('pth')
        if not os.path.exists(self.visual_log):
            os.makedirs(self.visual_log)
        if not os.path.exists('loss'):
            os.makedirs('loss')
        if not os.path.exists('result'):
            os.makedirs('result')
