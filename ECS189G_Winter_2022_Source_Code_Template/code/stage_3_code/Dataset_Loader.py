from code.base_class.dataset import dataset
import pickle
import matplotlib.pyplot as plt

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    cmap = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        for instance in data['train']:
            image_matrix = instance['image']
            image_label = instance['label']
            plt.imshow(image_matrix, cmap=self.cmap)
            plt.show()
            print(image_matrix)
            print(image_label)
            # remove the following "break" code if you would like to see more image in the training set
            break
            
        for instance in data['test']:
            image_matrix = instance['image']
            image_label = instance['label']
            plt.imshow(image_matrix, cmap=self.cmap)
            plt.show()
            print(image_matrix)
            print(image_label)
            # remove the following "break" code if you would like to see more image in the testing set
            break

        return {
            'train' : {
                'image' : [i['image'] for i in data['train']],
                'label' : [i['label'] for i in data['train']]
            },
            'test' : {
                'image' : [i['image'] for i in data['test']],
                'label' : [i['label'] for i in data['test']]
            }
        }