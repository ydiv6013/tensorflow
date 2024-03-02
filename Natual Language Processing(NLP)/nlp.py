import pandas as pd

textdatasetpath = "/Users/yogesh/Intelligent Vision/AML/Assessment1/flickr8k-dataset/flickr8k.TrainImages.txt"

# read the text files
with open(textdatasetpath ,"r") as file :
    lines = file.readlines()


image_names = []
strings = []
labels = []

for line in lines :
    # extract images names
    line = line.strip().split()
    image_name = line[0]
    image_names.append(image_name)
    # extract lables
    label = line[-1]
    labels.append(label)
    # extract strings :
    string = ' '.join(line[1:-1])
    strings.append(string)

dataframe = pd.DataFrame({"images" : image_names,
                        "strings": strings,
                        "labels": labels})

stringdf = dataframe["strings"]
print(dataframe["strings"])