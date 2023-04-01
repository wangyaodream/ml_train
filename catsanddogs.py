import os, shutil

from keras.preprocessing.image import ImageDataGenerator


original_dataset_dir = "/Users/wangyao/tmp/dogs-vs-cats/train/"
base_dir = '/Users/wangyao/tmp/cats_and_dogs_small'

class Handler:
    def __init__(self):
        self.run()


    def init_dir(self):
        print('start...')
        if os.path.exists(base_dir):
            return

        os.mkdir(base_dir)

        self.train_dir = os.path.join(base_dir, 'train')
        os.mkdir(self.train_dir)
        self.validation_dir = os.path.join(base_dir, 'validation') 
        os.mkdir(self.validation_dir)
        self.test_dir = os.path.join(base_dir, 'test')
        os.mkdir(self.test_dir)

        self.train_cats_dir = os.path.join(self.train_dir, 'cats')
        os.mkdir(self.train_cats_dir)
        self.train_dogs_dir = os.path.join(self.train_dir, 'dogs')
        os.mkdir(self.train_dogs_dir)

        self.validation_cats_dir = os.path.join(self.validation_dir, 'cats')
        os.mkdir(self.validation_cats_dir)
        self.validation_dogs_dir = os.path.join(self.validation_dir, 'dogs')
        os.mkdir(self.validation_dogs_dir)

        self.test_cats_dir = os.path.join(self.test_dir, 'cats')
        os.mkdir(self.test_cats_dir)
        self.test_dogs_dir = os.path.join(self.test_dir, 'dogs')
        os.mkdir(self.test_dogs_dir)

        # 将对应图片填充到各取样目录内

        # 将前1000张猫的图像复制到train_cats_dir中
        fnames = [f'cat.{i}.jpg' for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(self.train_cats_dir, fname)
            shutil.copyfile(src, dst)

        # 将500张猫的图片复制到validation_cats_dir
        fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(self.validation_cats_dir, fname)
            shutil.copyfile(src, dst)

        # 将接下来500张复制到测试集
        fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(self.test_cats_dir, fname)
            shutil.copyfile(src, dst)


        # 将1000张狗的图像复制到train_dogs_dir中
        fnames = [f'dog.{i}.jpg' for i in range(1000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(self.train_dogs_dir, fname)
            shutil.copyfile(src, dst)

        # 将接下来500张图像复制到validation_dogs_dir
        fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(self.validation_dogs_dir, fname)
            shutil.copyfile(src, dst)

        # 狗狗的测试集
        fnames = [f'dog.{i}.jpg' for i in range(1500, 2000)]
        for fname in fnames:
            src = os.path.join(original_dataset_dir, fname)
            dst = os.path.join(self.test_dogs_dir, fname)
            shutil.copyfile(src, dst)

        print("Done!")

    def run(self):
        self.init_dir()

        # 将素有图像乘以1/255缩放
        train_dategen = ImageDataGenerator(rescale=1./255)
        test_dategen = ImageDataGenerator(rescale=1./255)

        train_generator = train_dategen.flow_from_directory(
            self.train_dir,
            target_size=(150,150),
            batch_size=20,
            class_mode='binary'
                )

        validation_generator = test_dategen.flow_from_directory(
            self.validation_dir,
            target_size=(150,150),
            batch_size=20,
            class_mode='binary'
        )

def main():
    handle = Handler()
    handle.run()
    print("目录生成完毕!")


if __name__ == "__main__":
    main()

