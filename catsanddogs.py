import json
import os, shutil

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator


original_dataset_dir = "/Users/wangyao/tmp/dogs-vs-cats/train/"
base_dir = '/Users/wangyao/tmp/cats_and_dogs_small'

class Handler:
    def __init__(self):
        self.run()


    def init_dir(self):
        print('start...')
        if os.path.exists(base_dir):
            self.train_dir = os.path.join(base_dir, 'train')
            self.validation_dir = os.path.join(base_dir, 'validation') 
            self.test_dir = os.path.join(base_dir, 'test')
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

        # 构建网络
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.RMSprop(learning_rate=1e-4),
                      metrics=['accuracy'])

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

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=50
        )
        model.save('temp/cats_and_dogs_small_1.h5')
        with open("temp/result.json", 'w') as fp:
            fp.write(json.dumps(history.history))
        print("Done!")


def main():
    handle = Handler()
    handle.run()


if __name__ == "__main__":
    main()

