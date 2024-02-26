import os
from sklearn.model_selection import train_test_split
from shutil import copyfile

class GenerateDataset:
    def __init__(self) -> None:
        pass

    def generate_dataset(self):
        # Dataset paths
        path = "/Users/yogesh/pythoncode/datasets/pizza-samosa/"

        pizza_path = os.path.join(path,"pizza")
        samosa_path = os.path.join(path,"samosa")

        # Train set paths
        train_pizza_path = os.path.join(path, "train", "pizza")
        train_samosa_path = os.path.join(path, "train", "samosa")

        # Test set paths
        test_pizza_path = os.path.join(path, "test", "pizza")
        test_samosa_path = os.path.join(path, "test", "samosa")

        # Create directories if they don't exist
        os.makedirs(train_pizza_path, exist_ok=True)
        os.makedirs(train_samosa_path, exist_ok=True)
        os.makedirs(test_pizza_path, exist_ok=True)
        os.makedirs(test_samosa_path, exist_ok=True)

        # List all the images in pizza and samosa datasets
        all_pizza_images = [file for file in os.listdir(pizza_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        all_samosa_images = [file for file in os.listdir(samosa_path) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        # Split the pizza images into pizza training set and test sets
        pizza_train_images, pizza_test_images = train_test_split(all_pizza_images, test_size=0.30, random_state=42)
        samosa_train_images, samosa_test_images = train_test_split(all_samosa_images, test_size=0.30, random_state=42)

        # Copy pizza set images to pizza train set directory
        pizza_train_urls = [copyfile(os.path.join(pizza_path, image), os.path.join(train_pizza_path, image)) for image in pizza_train_images]

        # Copy samosa set images to samosa train set directory
        samosa_train_urls = [copyfile(os.path.join(samosa_path, image), os.path.join(train_samosa_path, image)) for image in samosa_train_images]

        # Copy pizza set images to pizza test set directory
        pizza_test_urls = [copyfile(os.path.join(pizza_path, image), os.path.join(test_pizza_path, image)) for image in pizza_test_images]

        # Copy samosa set images to samosa test set directory
        samosa_test_urls = [copyfile(os.path.join(samosa_path, image), os.path.join(test_samosa_path, image)) for image in samosa_test_images]

        return pizza_train_urls, pizza_test_urls, samosa_train_urls, samosa_test_urls

if __name__ == "__main__":
    gd = GenerateDataset()
    gd.generate_dataset()