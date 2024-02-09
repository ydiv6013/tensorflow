import os
from shutil import copyfile
from sklearn.model_selection import train_test_split

class DatsetGenerator :
    def __init__(self) -> None:
        pass
    def generate_dataset(self,dataset_path) :

        # happy dataset path
        happy_dataset = dataset_path + "/Happy"
        sad_dataset = dataset_path + "/Sad"

        # set train set and test set path 
        happy_train_path = dataset_path + "/training_set" + "/Happy"
        sad_train_path = dataset_path + "/training_set" + "/Sad"
        
        happy_test_path = dataset_path + "/test_set" + "/Happy"
        sad_test_path = dataset_path + "/test_set" + "/Sad"
        
        # create the directories if they doesnt exist
        os.makedirs(happy_train_path, exist_ok=True)
        os.makedirs(sad_train_path , exist_ok= True)

        os.makedirs(happy_test_path, exist_ok=True)
        os.makedirs(sad_test_path , exist_ok= True)

        # list all the images of happy dataset
        all_happy_images = [file for file in os.listdir(happy_dataset) if file.lower().endswith(('.jpg', '.jpeg','.png','.webp'))]
        all_sad_images = [file for file in os.listdir(sad_dataset) if file.lower().endswith(('.jpg', '.jpeg','.png','.webp'))]
        

        # split the happy images into happy trainig set and test sets
        happy_train_images , happy_test_images = train_test_split(all_happy_images , test_size= 0.20 ,random_state= 42)
        sad_train_images ,sad_test_images = train_test_split(all_sad_images ,test_size=0.20 ,random_state= 42)

        # copy Happy set images to  happy train set directory
        happy_train_url = []
        sad_train_url = []
        happy_test_url = []
        sad_test_url = []
        for image in happy_train_images :
            source_path = os.path.join(happy_dataset,image)
            destination_path = os.path.join(happy_train_path,image)
            happy_train_url.append(copyfile(source_path,destination_path))

        # copy Sad set images to  sad train set directory

        for image in sad_train_images :
            source_path = os.path.join(sad_dataset,image)
            destination_path = os.path.join(sad_train_path,image)
            sad_train_url.append(copyfile(source_path,destination_path))

         # copy Happy set images to  happy test set directory

        for image in happy_test_images :
            source_path = os.path.join(happy_dataset,image)
            destination_path = os.path.join(happy_test_path,image)
            happy_test_url.append(copyfile(source_path,destination_path))

        

         # copy Sad set images  to  Sad test set directory

        for image in sad_test_images :
            source_path = os.path.join(sad_dataset,image)
            destination_path = os.path.join(sad_test_path,image)
            sad_test_url.append(copyfile(source_path,destination_path))

        return happy_train_url ,happy_test_url,sad_train_url,sad_test_url

    