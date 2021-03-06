# EE 298-F Final Project
# Miranda, Neil Jon Louie P.
# 2007-46489

import argparse, os
from time import time
from keras.callbacks import ModelCheckpoint, TensorBoard

from dataset_generator import DatasetGenerator as Generator
import model_builder

import numpy as np
import cv2

# Settings
image_size = (608, 1632, 3)
batch_size = 8
epochs = 20
dataset_root_dir = 'dataset-sample/'
trn_list = 'list/train.txt'
val_list = 'list/val.txt'
colors = [[255, 0, 0], [0, 255, 255], [0, 0, 255], [255, 255, 0]]

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('option', help='test|train')
parser.add_argument('--weights_file', help='Weights file (.hdf5) to be used for training or testing')
parser.add_argument('--num_epochs', type=int, help='Number of epochs to be run during training')
parser.add_argument('--input_file', help='Path to image file to be used')
args = parser.parse_args()

# Test the trained model
if args.option == 'test':
    if args.weights_file is None:
        print('Error: Weights file needed for prediction. Please provide a weights file using the --weights_file argument.')
        quit()
    
    if args.input_file is None:
        print('Error: Input file needed for prediction. Please provide an input file using the --input_file argument.')
        quit()

    if not os.path.exists(args.input_file):
        print ('File %s not found.' % args.input_file)
        quit()

    model = model_builder.build_model(image_size)
    model.load_weights(args.weights_file)

    image = cv2.imread(args.input_file)
    image = cv2.resize(image, dsize=(image_size[1], image_size[0]))
    image_array = np.array([image,])
    pred = model.predict(image_array)
    pred = np.argmax(pred, axis=3)
    pred = np.squeeze(pred)
    
    for i in range(len(colors)):
        image[pred == i + 1] = colors[i]
    cv2.imwrite('pred.png', image)

# Train the model
elif args.option == 'train':
    trn_generator = Generator(dataset_root_dir, trn_list, batch_size)
    val_generator = Generator(dataset_root_dir, val_list, batch_size)

    # model = model_builder.build_model((590, 1640, 3))
    model = model_builder.build_model(image_size)
    model.summary()
    if not args.weights_file is None:
        model.load_weights(args.weights_file)

    if not args.num_epochs is None:
        epochs = args.num_epochs

    tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))

    filepath = 'weights-{epoch:02d}-{val_acc:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
    model.fit_generator(trn_generator, epochs=epochs, verbose=1, validation_data=val_generator, callbacks=[tensorboard, checkpoint])

else:
    print('Unknown option "%s". Valid options: test, train' % args.option)
