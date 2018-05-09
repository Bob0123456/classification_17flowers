import random
import os
from dataset import Dataset
from model import CNN2D
from keras.optimizers import SGD

def main():
    batch_size = 128
    image_shape = (80, 80, 3)

    # create dataset
    data = Dataset(batch_size, image_shape)
    

    # create model
    model = CNN2D(len(data.classes), image_shape)
    optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    # load weights
    model.load_weights('./log/retrain/weights.hdf5')

    # # choice a random input
    # sample = random.choice(data.test)
    # print("\n-------------------------\n Sample:")
    # print(sample)
    # image = data.load_sample(sample)

    # # output the true result 
    # sample_result = data.get_class_one_hot(sample[0])
    # print("\n-------------------------\n True Result:")
    # print(sample_result)
    # print(sample_result.max())
    # print(sample_result.argmax())


    # # get predict result
    # predict_result = model.predict(image)[0]

    # print("\n-------------------------\n Predict Result:")
    # print(predict_result)
    # print(predict_result.max())
    # print(predict_result.argmax())
    
    train_dataset_count = 0
    train_true_count = 0
    
    for sample in data.train:
        image = data.load_sample(sample)
        sample_result = data.get_class_one_hot(sample[0])
        predict_result = model.predict(image)[0]
        if sample_result.argmax() == predict_result.argmax():
            train_true_count = train_true_count + 1
        train_dataset_count = train_dataset_count + 1
    print(' Train accuary: {0:4d}  /  {1:4d}  =  {2:6f} \n'.format(train_true_count, train_dataset_count, train_true_count / train_dataset_count))

    test_dataset_count = 0
    test_true_count = 0

    for sample in data.test:
        image = data.load_sample(sample)
        sample_result = data.get_class_one_hot(sample[0])
        predict_result = model.predict(image)[0]
        if sample_result.argmax() == predict_result.argmax():
            test_true_count = test_true_count + 1
        test_dataset_count = test_dataset_count + 1
    
    print(' Test accuary: {0:4d}  /  {1:4d}  =  {2:6f} \n'.format(test_true_count, test_dataset_count, test_true_count / test_dataset_count))

    true_count = train_true_count + test_true_count
    dataset_count = train_dataset_count + test_dataset_count
    print(' All accuary: {0:4d}  /  {1:4d}  =  {2:6f} \n'.format(true_count, dataset_count, true_count / dataset_count))


if __name__ == '__main__':
    main()
