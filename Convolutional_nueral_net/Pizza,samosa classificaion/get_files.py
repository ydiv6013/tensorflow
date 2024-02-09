import os


class GetFiles :
    def __init__(self) -> None:
        pass

    def get_files(self,filepath):

        filepath = filepath
        dirpath,dirname,filename = os.walk(filepath) 
       

        dir1 = str(dirpath) + "/" + "samosa"
        dir2 = str(dirpath) + "/" + "pizza"

        print(dir1)
        print(os.path)




if __name__ == "__main__" :
    
    image_datasets_path  = "/Users/yogesh/pythoncode/datasets/pizza-samosa/"

    gf = GetFiles()
    gf.get_files(filepath=image_datasets_path)
