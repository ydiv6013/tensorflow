import os
import cv2 as cv

path = "/Users/yogesh/pythoncode/datasets/pizza-samosa/pizza/"
image = "2965.jpg"
url = str(path) + str(image)

print(url)

img = cv.imread(url)
"""
if img is None :
    print("Opps something went wrong")
else :"""
cv.imshow("orignal image",image)
cv.waitKey()
cv.destroyAllWindows()

class GetFiles :
    def __init__(self) -> None:
        pass

    def get_files(self,filepath):

        filepath = filepath
        dirpath,dirname,filename = os.walk(filepath) 
        print("name of the directory is : \n",dirname)
       

        dir1 = str(dirpath) + "/" + "samosa"
        dir2 = str(dirpath) + "/" + "pizza"

        print(dir1)
        print(os.path)
        
        for files in filename :
            print(files)
            urls1 = str(dir1) + str(files)
        
        print("\n urls are : \n" ,urls1)
        print("\n Files are :\n" ,files)
        





if __name__ == "__main__" :
    
    image_datasets_path  = "/Users/yogesh/pythoncode/datasets/pizza-samosa/"

    gf = GetFiles()
    #gf.get_files(filepath=image_datasets_path)
