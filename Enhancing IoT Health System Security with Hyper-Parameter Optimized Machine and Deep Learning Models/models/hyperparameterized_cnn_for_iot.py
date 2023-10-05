# -*- coding: utf-8 -*-
"""Hyperparameterized CNN for IOT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e4MMoOsSjMAqzqnjXnr0P1LN1QsNuXpb
"""

# Commented out IPython magic to ensure Python compatibility.
"""
   Compile, Build, and Perform Hyperparameters Optimization in order to Build the Deep Learning CNN, LSTM, and
   Attention Based CNN_LSTM Models:
"""

"""
   Model 4 - Convolutional Neural Network (CNN)

   We will apply the CNN algorithm to our data to generate prediction results. First, we need to reshape our data for CNN.
   We will use 1-dimensional CNN for our model, reshaping our data as per the dimensions of our CNN
"""

"""Note that the timeseries data used here are univariate, meaning we only have one channel per timeseries example.
We will therefore transform the timeseries into a multivariate one with one channel using a simple
reshaping via numpy. This will allow us to construct a model that is easily applicable to multivariate time series"""

# Reshape the Data:
X_train = X_train_smote.reshape(len(X_train_smote), X_train_smote.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
X_train.shape, X_test.shape

# %load_ext tensorboard

def build_model(learning_rate=3e-3, input_shape=(13, 1)):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=6, activation='relu',
                    padding='same', input_shape=(13, 1)))
    model.add(BatchNormalization())

    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                    padding='same', input_shape=(13, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                    padding='same', input_shape=(13, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax'))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy', f1_m, precision_m, recall_m])
    return model

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)
model_checkpoint_cb = keras.callbacks.ModelCheckpoint("CNN.h5", save_best_only=True)
run_index = 1 # increment every time you train the model
run_logdir = os.path.join(os.curdir, "my_CNN_logs", "run_bn_{:03d}".format(run_index))
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
callbacks = [early_stopping_cb, model_checkpoint_cb, tensorboard_cb]

"""
    Hyperparameters Optimization Using Randomized Search Algorithm
"""
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

param_distribs = {
    #"n_hidden": [1, 2, 3],
    #"n_filters": np.arange(1, 100)               .tolist(),
    "learning_rate": reciprocal(3e-4, 3e-2)      .rvs(1000).tolist(),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train_smote, epochs=10,
                  validation_split=0.3,
                  callbacks=callbacks)
"""
    Obtain the best Parameters:
"""
print(rnd_search_cv.best_params_)

"""
    Obtain the best Score:
"""
print(rnd_search_cv.best_score_)

rnd_search_cv.score(X_test, y_test)

"""
    Obtain the best Trained Model and use it for Evaluation:
"""
model = rnd_search_cv.best_estimator_.model
model.evaluate(X_test, y_test)

"""
   Make Predictions with the trained Model:
"""
# making predictions on test set
start = time.time()

y_pred_CNN = model.predict(X_test)

end = time.time()
print('Time Taken: %.3f seconds' % (end-start))

print('Accuracy:', accuracy_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1)))
print('F1 score:', f1_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1), average='weighted'))
print('Recall:', recall_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1), average='weighted'))
print('Precision:', precision_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1), average='weighted'))
print('\n clasification report:\n', classification_report(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1)))
print('\n confussion matrix:\n', Error_matrix(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1)))

performance_all['CNN_Model'] = accuracy_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1)), f1_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1), average='weighted'), recall_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1), average='weighted'), precision_score(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1), average='weighted')

#model2 = clf
#all_models.append(model2)

# Plot Model's Error Matrix:

def plot_Error_matrix(cm, classes,
                          normalize=False,
                          title='Error matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Error matrix")
    else:
        print('Error matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 20, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label', weight='bold').set_fontsize('18')
    plt.xlabel('Predicted Label', weight='bold').set_fontsize('18')
    plt.savefig('Error Matrix for CNN Model.png')

cf_matrix = Error_matrix(y_test.argmax(axis=1), y_pred_CNN.argmax(axis=1))
plt.figure(figsize=(15, 12))
plot_Error_matrix(cf_matrix, classes=['Normal', 'Backdoor Attack', 'injection Attack', 'password Attack', 'ddos Attack', 'ransomware Attack', 'xss Attack', 'scanning Attack'],
                      normalize = True, title='Error Matrix With Normalization - CNN Model')
plt.show()

model.summary()