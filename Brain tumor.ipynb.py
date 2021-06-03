{
 "cells": [
  {
   "cell_type": "code"
   "execution_count": 1, "metadata": {},"outputs": [
    {
     "name": "stderr", "output_type": "stream", "text": [ "/usr/local/lib/python2.7/dist-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n", "  from ._conv import register_converters as _register_converters\n",   "Using TensorFlow backend.\n"  ]   } ],
   "source": [
    "import tensorflow as tf\n",
    "import keras\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Conv2D\n",
    "from keras.layers import MaxPooling2D\n",
    "from keras.layers import Flatten\n",
    "from keras.layers import Dense"
   ]
  },
  { "cell_type": "code", "execution_count": 2 "metadata": {},  "outputs": [],  "source": [ "classifier = Sequential()\n", "\n", "classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))"  ] },
  {
   "cell_type": "code", "execution_count": 3, "metadata": {}, "outputs": [
    {  "name": "stdout", "output_type": "stream", "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_1 (Conv2D)            (None, 62, 62, 32)        896       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 31, 31, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 29, 29, 32)        9248      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 6272)              0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 128)               802944    \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 1)                 129       \n",
      "=================================================================\n",
      "Total params: 813,217\n",
      "Trainable params: 813,217\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "classifier.add(MaxPooling2D(pool_size = (2, 2)))\n",
    "\n",
    "classifier.add(Conv2D(32, (3, 3), activation = 'relu'))\n",
    "classifier.add(MaxPooling2D(pool_size = (2, 2)))\n",
    "\n", "classifier.add(Flatten())\n", "\n", "classifier.add(Dense(activation = 'relu',units=128))\n", "classifier.add(Dense(activation = 'sigmoid',units=1))\n", "\n", "classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])\n", "\n", "classifier.summary()" ]  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "train_datagen = ImageDataGenerator(rescale = 1./255,\n",
    "                                   shear_range = 0.2,\n",
    "                                   zoom_range = 0.2,\n",
    "                                   horizontal_flip = True)\n",
    "\n",
    "test_datagen = ImageDataGenerator(rescale = 1./255)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/telraswa/Desktop/Swapnil/manju_project/Brain_tumor\n"
     ]
    }
   ],
   "source": [
    "import os \n",
    "os.getcwd()\n",
    "os.chdir('/home/telraswa/Desktop/Swapnil/manju_project/Brain_tumor')\n",
    "print(os.getcwd())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 22 images belonging to 2 classes.\n",
      "Found 7 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "training_set = train_datagen.flow_from_directory('/home/telraswa/Desktop/Swapnil/manju_project/Brain_tumor/train/',\n",
    "                                                 target_size = (64, 64),\n",
    "                                                 batch_size = 32,\n",
    "                                                 class_mode = 'binary')\n",
    "\n",
    "test_set = test_datagen.flow_from_directory('/home/telraswa/Desktop/Swapnil/manju_project/Brain_tumor/test/',\n",
    "                                            target_size = (64, 64),\n",
    "                                            batch_size = 32,\n",
    "                                            class_mode = 'binary')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "1/1 [==============================] - 0s 265ms/step - loss: 0.6989 - acc: 0.4545 - val_loss: 0.6783 - val_acc: 0.5714\n",
      "Epoch 2/100\n",
      "1/1 [==============================] - 0s 85ms/step - loss: 0.7245 - acc: 0.5909 - val_loss: 0.6657 - val_acc: 0.5714\n",
      "Epoch 3/100\n",
      "1/1 [==============================] - 0s 92ms/step - loss: 0.6643 - acc: 0.5909 - val_loss: 0.6835 - val_acc: 0.7143\n",
      "Epoch 4/100\n",
      "1/1 [==============================] - 0s 96ms/step - loss: 0.6570 - acc: 0.6364 - val_loss: 0.6919 - val_acc: 0.4286\n",
      "Epoch 5/100\n",
      "1/1 [==============================] - 0s 95ms/step - loss: 0.6660 - acc: 0.6364 - val_loss: 0.6866 - val_acc: 0.4286\n",
      "Epoch 6/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.6521 - acc: 0.5909 - val_loss: 0.6723 - val_acc: 0.4286\n",
      "Epoch 7/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.6150 - acc: 0.6818 - val_loss: 0.6587 - val_acc: 0.5714\n",
      "Epoch 8/100\n",
      "1/1 [==============================] - 0s 104ms/step - loss: 0.6299 - acc: 0.6818 - val_loss: 0.6565 - val_acc: 0.7143\n",
      "Epoch 9/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.6036 - acc: 0.6364 - val_loss: 0.6674 - val_acc: 0.4286\n",
      "Epoch 10/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.5500 - acc: 0.7727 - val_loss: 0.6977 - val_acc: 0.4286\n",
      "Epoch 11/100\n",
      "1/1 [==============================] - 0s 103ms/step - loss: 0.5731 - acc: 0.7727 - val_loss: 0.7399 - val_acc: 0.4286\n",
      "Epoch 12/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.5595 - acc: 0.7727 - val_loss: 0.7293 - val_acc: 0.4286\n",
      "Epoch 13/100\n",
      "1/1 [==============================] - 0s 104ms/step - loss: 0.5355 - acc: 0.7727 - val_loss: 0.7543 - val_acc: 0.5714\n",
      "Epoch 14/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.4347 - acc: 0.8182 - val_loss: 0.8199 - val_acc: 0.5714\n",
      "Epoch 15/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.4917 - acc: 0.8182 - val_loss: 0.8882 - val_acc: 0.5714\n",
      "Epoch 16/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.4553 - acc: 0.8182 - val_loss: 0.9052 - val_acc: 0.5714\n",
      "Epoch 17/100\n",
      "1/1 [==============================] - 0s 106ms/step - loss: 0.4476 - acc: 0.8636 - val_loss: 0.9836 - val_acc: 0.5714\n",
      "Epoch 18/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.4721 - acc: 0.7727 - val_loss: 0.9389 - val_acc: 0.5714\n",
      "Epoch 19/100\n",
      "1/1 [==============================] - 0s 103ms/step - loss: 0.4296 - acc: 0.8636 - val_loss: 1.0669 - val_acc: 0.5714\n",
      "Epoch 20/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.4639 - acc: 0.8182 - val_loss: 1.0738 - val_acc: 0.5714\n",
      "Epoch 21/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.3790 - acc: 0.8182 - val_loss: 0.9798 - val_acc: 0.5714\n",
      "Epoch 22/100\n",
      "1/1 [==============================] - 0s 105ms/step - loss: 0.3320 - acc: 0.8182 - val_loss: 1.0362 - val_acc: 0.5714\n",
      "Epoch 23/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.3830 - acc: 0.7727 - val_loss: 1.1098 - val_acc: 0.5714\n",
      "Epoch 24/100\n",
      "1/1 [==============================] - 0s 103ms/step - loss: 0.2790 - acc: 0.9545 - val_loss: 1.2193 - val_acc: 0.5714\n",
      "Epoch 25/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.3339 - acc: 0.9091 - val_loss: 1.1455 - val_acc: 0.5714\n",
      "Epoch 26/100\n",
      "1/1 [==============================] - 0s 107ms/step - loss: 0.3693 - acc: 0.8636 - val_loss: 1.0581 - val_acc: 0.5714\n",
      "Epoch 27/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.2317 - acc: 0.9091 - val_loss: 1.0835 - val_acc: 0.5714\n",
      "Epoch 28/100\n",
      "1/1 [==============================] - 0s 95ms/step - loss: 0.3524 - acc: 0.8182 - val_loss: 1.1099 - val_acc: 0.5714\n",
      "Epoch 29/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.2668 - acc: 0.9545 - val_loss: 1.0763 - val_acc: 0.7143\n",
      "Epoch 30/100\n",
      "1/1 [==============================] - 0s 96ms/step - loss: 0.1509 - acc: 1.0000 - val_loss: 1.0829 - val_acc: 0.7143\n",
      "Epoch 31/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.2874 - acc: 0.9091 - val_loss: 1.1191 - val_acc: 0.5714\n",
      "Epoch 32/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.2976 - acc: 0.9091 - val_loss: 1.2024 - val_acc: 0.5714\n",
      "Epoch 33/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.3284 - acc: 0.8636 - val_loss: 1.1639 - val_acc: 0.7143\n",
      "Epoch 34/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.2315 - acc: 0.9545 - val_loss: 1.2063 - val_acc: 0.7143\n",
      "Epoch 35/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.2209 - acc: 0.8636 - val_loss: 1.3964 - val_acc: 0.4286\n",
      "Epoch 36/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1859 - acc: 0.9545 - val_loss: 1.6606 - val_acc: 0.4286\n",
      "Epoch 37/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1837 - acc: 0.9545 - val_loss: 1.4230 - val_acc: 0.5714\n",
      "Epoch 38/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.2424 - acc: 0.9545 - val_loss: 1.3140 - val_acc: 0.7143\n",
      "Epoch 39/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.2681 - acc: 0.9545 - val_loss: 1.3749 - val_acc: 0.7143\n",
      "Epoch 40/100\n",
      "1/1 [==============================] - 0s 105ms/step - loss: 0.2435 - acc: 0.9091 - val_loss: 1.7977 - val_acc: 0.4286\n",
      "Epoch 41/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.2477 - acc: 0.9091 - val_loss: 1.5881 - val_acc: 0.4286\n",
      "Epoch 42/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.1781 - acc: 0.9091 - val_loss: 1.3263 - val_acc: 0.7143\n",
      "Epoch 43/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1496 - acc: 0.9091 - val_loss: 1.3206 - val_acc: 0.7143\n",
      "Epoch 44/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1921 - acc: 0.9091 - val_loss: 1.4173 - val_acc: 0.5714\n",
      "Epoch 45/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.2026 - acc: 0.9091 - val_loss: 1.9731 - val_acc: 0.4286\n",
      "Epoch 46/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.2705 - acc: 0.9091 - val_loss: 1.9763 - val_acc: 0.4286\n",
      "Epoch 47/100\n",
      "1/1 [==============================] - 0s 103ms/step - loss: 0.2917 - acc: 0.9091 - val_loss: 1.5498 - val_acc: 0.4286\n",
      "Epoch 48/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.1798 - acc: 0.9091 - val_loss: 1.3730 - val_acc: 0.7143\n",
      "Epoch 49/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1683 - acc: 0.9091 - val_loss: 1.3851 - val_acc: 0.7143\n",
      "Epoch 50/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.2392 - acc: 0.8636 - val_loss: 1.4212 - val_acc: 0.7143\n",
      "Epoch 51/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.1434 - acc: 0.9545 - val_loss: 1.6520 - val_acc: 0.4286\n",
      "Epoch 52/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1493 - acc: 0.9545 - val_loss: 1.8999 - val_acc: 0.4286\n",
      "Epoch 53/100\n",
      "1/1 [==============================] - 0s 103ms/step - loss: 0.1785 - acc: 0.9545 - val_loss: 1.9982 - val_acc: 0.4286\n",
      "Epoch 54/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.2223 - acc: 0.9091 - val_loss: 1.7886 - val_acc: 0.4286\n",
      "Epoch 55/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1324 - acc: 0.9545 - val_loss: 1.5349 - val_acc: 0.7143\n",
      "Epoch 56/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.1228 - acc: 0.9545 - val_loss: 1.5115 - val_acc: 0.7143\n",
      "Epoch 57/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.1828 - acc: 0.9545 - val_loss: 1.5495 - val_acc: 0.7143\n",
      "Epoch 58/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.2627 - acc: 0.9091 - val_loss: 1.7841 - val_acc: 0.5714\n",
      "Epoch 59/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.1456 - acc: 0.9091 - val_loss: 2.0874 - val_acc: 0.4286\n",
      "Epoch 60/100\n",
      "1/1 [==============================] - 0s 96ms/step - loss: 0.1528 - acc: 0.9545 - val_loss: 2.1812 - val_acc: 0.4286\n",
      "Epoch 61/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.2300 - acc: 0.9091 - val_loss: 1.9207 - val_acc: 0.4286\n",
      "Epoch 62/100\n",
      "1/1 [==============================] - 0s 119ms/step - loss: 0.2139 - acc: 0.9545 - val_loss: 1.5568 - val_acc: 0.5714\n",
      "Epoch 63/100\n",
      "1/1 [==============================] - 0s 90ms/step - loss: 0.1709 - acc: 0.9091 - val_loss: 1.4649 - val_acc: 0.7143\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 64/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.2230 - acc: 0.9091 - val_loss: 1.4705 - val_acc: 0.7143\n",
      "Epoch 65/100\n",
      "1/1 [==============================] - 0s 105ms/step - loss: 0.3411 - acc: 0.9091 - val_loss: 1.4961 - val_acc: 0.7143\n",
      "Epoch 66/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.1426 - acc: 0.9545 - val_loss: 1.7115 - val_acc: 0.4286\n",
      "Epoch 67/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1642 - acc: 0.9091 - val_loss: 1.9849 - val_acc: 0.4286\n",
      "Epoch 68/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1722 - acc: 0.9545 - val_loss: 2.0723 - val_acc: 0.4286\n",
      "Epoch 69/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.2271 - acc: 0.9091 - val_loss: 1.8962 - val_acc: 0.4286\n",
      "Epoch 70/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.1398 - acc: 0.9545 - val_loss: 1.6629 - val_acc: 0.4286\n",
      "Epoch 71/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1588 - acc: 0.9545 - val_loss: 1.4951 - val_acc: 0.7143\n",
      "Epoch 72/100\n",
      "1/1 [==============================] - 0s 102ms/step - loss: 0.1285 - acc: 0.9545 - val_loss: 1.4719 - val_acc: 0.7143\n",
      "Epoch 73/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.1986 - acc: 0.9545 - val_loss: 1.4974 - val_acc: 0.7143\n",
      "Epoch 74/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.1906 - acc: 0.9545 - val_loss: 1.5856 - val_acc: 0.5714\n",
      "Epoch 75/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.1580 - acc: 0.9545 - val_loss: 1.7762 - val_acc: 0.5714\n",
      "Epoch 76/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1081 - acc: 1.0000 - val_loss: 2.0252 - val_acc: 0.4286\n",
      "Epoch 77/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.1878 - acc: 0.9091 - val_loss: 2.0940 - val_acc: 0.4286\n",
      "Epoch 78/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1409 - acc: 0.9545 - val_loss: 2.0419 - val_acc: 0.4286\n",
      "Epoch 79/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.0918 - acc: 1.0000 - val_loss: 1.9874 - val_acc: 0.4286\n",
      "Epoch 80/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1975 - acc: 0.9091 - val_loss: 1.8914 - val_acc: 0.5714\n",
      "Epoch 81/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1306 - acc: 0.9545 - val_loss: 1.8899 - val_acc: 0.5714\n",
      "Epoch 82/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1092 - acc: 0.9545 - val_loss: 1.9667 - val_acc: 0.5714\n",
      "Epoch 83/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1039 - acc: 0.9545 - val_loss: 2.0630 - val_acc: 0.4286\n",
      "Epoch 84/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.0937 - acc: 0.9545 - val_loss: 2.1282 - val_acc: 0.4286\n",
      "Epoch 85/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1057 - acc: 0.9091 - val_loss: 2.1899 - val_acc: 0.4286\n",
      "Epoch 86/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.0906 - acc: 1.0000 - val_loss: 2.1741 - val_acc: 0.4286\n",
      "Epoch 87/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.0949 - acc: 0.9545 - val_loss: 2.1312 - val_acc: 0.4286\n",
      "Epoch 88/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.0840 - acc: 0.9545 - val_loss: 2.0405 - val_acc: 0.5714\n",
      "Epoch 89/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.0966 - acc: 1.0000 - val_loss: 2.0102 - val_acc: 0.5714\n",
      "Epoch 90/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.1032 - acc: 0.9091 - val_loss: 1.9665 - val_acc: 0.5714\n",
      "Epoch 91/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.0516 - acc: 1.0000 - val_loss: 1.9753 - val_acc: 0.5714\n",
      "Epoch 92/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1111 - acc: 0.9545 - val_loss: 1.9438 - val_acc: 0.5714\n",
      "Epoch 93/100\n",
      "1/1 [==============================] - 0s 97ms/step - loss: 0.1661 - acc: 0.9545 - val_loss: 1.9585 - val_acc: 0.5714\n",
      "Epoch 94/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1103 - acc: 0.9545 - val_loss: 1.9688 - val_acc: 0.5714\n",
      "Epoch 95/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.0623 - acc: 0.9545 - val_loss: 1.9698 - val_acc: 0.5714\n",
      "Epoch 96/100\n",
      "1/1 [==============================] - 0s 99ms/step - loss: 0.1440 - acc: 0.9545 - val_loss: 2.0926 - val_acc: 0.5714\n",
      "Epoch 97/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.0680 - acc: 0.9545 - val_loss: 2.1961 - val_acc: 0.4286\n",
      "Epoch 98/100\n",
      "1/1 [==============================] - 0s 98ms/step - loss: 0.0230 - acc: 1.0000 - val_loss: 2.2949 - val_acc: 0.4286\n",
      "Epoch 99/100\n",
      "1/1 [==============================] - 0s 100ms/step - loss: 0.1330 - acc: 0.9091 - val_loss: 2.4832 - val_acc: 0.4286\n",
      "Epoch 100/100\n",
      "1/1 [==============================] - 0s 101ms/step - loss: 0.0841 - acc: 0.9545 - val_loss: 2.4504 - val_acc: 0.4286\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x7f2ac2703cd0>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "classifier.fit_generator(training_set, steps_per_epoch=None, epochs=100, verbose=1, callbacks=None, validation_data=test_set, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": \n",
      "text/plain": [
       "<PIL.Image.Image image mode=RGB size=64x64 at 0x7F2AE3566E50>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "from keras.preprocessing import image\n",
    "test_image = image.load_img('/home/telraswa/Desktop/Swapnil/manju_project/TestImages/brain-tumors-fig2_large.jpg', target_size = (64, 64))\n",
    "test_image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[[146., 146., 146.],\n",
       "         [ 59.,  59.,  59.],\n",
       "         [ 10.,  10.,  10.],\n",
       "         ...,\n",
       "         [ 13.,  13.,  13.],\n",
       "  [ 13.,  13.,  13.],\n", " [ 14.,  14.,  14.]],\n", "\n",        "        [ 65.,  65.,  65.],\n",
      "  [101., 101., 101.],\n",
       "         [ 15.,  15.,  15.],\n",
       "         ...,\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.]],\n",
       "\n",
       "        [[ 14.,  14.,  14.],\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 10.,  10.,  10.],\n",
       "         ...,\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.]],\n",
       "\n",
       "        ...,\n",
       "\n",
       "        [[120., 120., 120.],\n",
       "         [  5.,   5.,   5.],\n",
       "         [  9.,   9.,   9.],\n",
       "         ...,\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.]],\n",
       "\n",
       "        [[174., 174., 174.],\n",
       "         [ 85.,  85.,  85.],\n",
       "         [193., 193., 193.],\n",
       "         ...,\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.]],\n",
       "\n",
       "        [[ 13.,  13.,  13.],\n",
       "         [ 13.,  13.,  13.],\n",
       "         [ 13.,  13.,  13.],\n",
       "         ...,\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.],\n",
       "         [ 11.,  11.,  11.]]]], dtype=float32)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_image = image.img_to_array(test_image)\n",
    "\n",
    "test_image = np.expand_dims(test_image, axis = 0)\n",
    "test_image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.]], dtype=float32)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "result = classifier.predict(test_image)\n",
    "result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'Benign': 0, 'Malignant': 1}"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_set.class_indices"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Detected tumor type is Malignent\n"
     ]
    }
   ],
   "source": [
    "if result[0][0] == 0:\n",
    "    prediction = 'Benign'\n",
    "else:\n",
    "    prediction = 'Malignent'\n",
    "print(\"Detected tumor type is %s\"%prediction)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
