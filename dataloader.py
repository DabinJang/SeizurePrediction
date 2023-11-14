import numpy as np
from tensorflow.keras.utils import Sequence
from read_dataset import GetBatchIndexes, LoadDataset, Interval2Segments_v2, Segments2Data_v2

class DataLoader(Sequence):
    def __init__(self, type_1_data, type_2_data, type_3_data, batch_size, shuffle=True):
        self.type_1_data = type_1_data
        self.type_2_data = type_2_data
        self.type_3_data = type_3_data
        self.shuffle = shuffle
        self.batch_size = batch_size
        
        self.x = self.sampled_data()
        self.indices = np.arange(len(self.x))

        if self.shuffle == True:
            np.random.shuffle(self.indices)
        
    def sampled_data(self, sample_rate=[1, .5, .1]):
        type_1_sampled_len = int(len(self.type_1_data)*sample_rate[0])
        type_2_sampled_len = int(len(self.type_2_data)*sample_rate[1])
        type_3_sampled_len = int(len(self.type_3_data)*sample_rate[2])

        type_1_sampled = self.type_1_data[np.random.choice(len(self.type_1_data), type_1_sampled_len, replace=False)]
        type_2_sampled = self.type_2_data[np.random.choice(len(self.type_2_data), type_2_sampled_len, replace=False)]
        type_3_sampled = self.type_3_data[np.random.choice(len(self.type_3_data), type_3_sampled_len, replace=False)]
        
        x = np.concatenate([type_1_sampled,type_2_sampled,type_3_sampled], axis=0)
        return x
    
    def __len__(self):
	    return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        input_seg = [self.x[i] for i in indices]
        batch_x, batch_y = Segments2Data_v2(input_seg)
        print(batch_x.shape, batch_y.shape)
        return batch_x, batch_y
    
    def on_epoch_end(self):
        self.x = self.sampled_data()
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            

if __name__=="__main__":
    pass