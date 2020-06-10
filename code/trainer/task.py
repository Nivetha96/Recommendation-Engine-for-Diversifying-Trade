from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import argparse
from keras import callbacks
from tensorflow.python.lib.io import file_io

def main(job_dir,**args):

    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'

    data = pd.read_csv('gs://datathon_new/finaldata.csv')

    data = data[data['Year']==2017]

    data = data.drop(['Population_x','Population_y','GDP.2_x','GDP.1_y','GDP.2_y','Share of_x','Share of_y','Comoroslang_off','Comoroslang_Ethiopiano','Colombiaony','ComorosColombia','curColombia','Colombia45','smctry','Frequency Code','Frequency','Indicator Category','Indicator Code','Indicator','Reporting Economy Code','Partner Economy Code','Product/Sector Classification Code','Period Code','GDP.1_x','Value Flag Code','Value Flag','Text Value','iso_o','iso_d','disTurkeyks and Caicos Islandsp','distw','distwces','Country_x'],axis=1)

    data['Product Group'] = (data['Product/Sector Code'] >= 50) & (data['Product/Sector Code'] <= 67)

    le = preprocessing.LabelEncoder()

    #data['Partner Economy Label'] = le.fit_transform(data['Partner Economy'])
    #data['Reporting Economy Label'] = le.fit_transform(data['Reporting Economy'])
    #data['Product Sector Label'] = le.fit_transform(data['Product/Sector'])
    data['Product Group Label'] = le.fit_transform(data['Product Group'])
    data['Y Label'] = le.fit_transform(data['Y'])

    data= data[data['Product Group Label'] == 1]

    final_df = data[['Value','Origin output','#trading partners','distance between origin and destination','Year','Reporting Economy GDP','Partner Economy GDP','FTA','import_duty']]
                    #,'colonised(1/0)','gdp of source']]

    final_df=final_df.fillna(0)
    final_df['Scaled Origin Output'] = StandardScaler().fit_transform(final_df[['Origin output']])
    final_df['scaled distance between origin and destination'] = StandardScaler().fit_transform(final_df[['distance between origin and destination']])
    final_df['Scaled Reporting Economy GDP'] = StandardScaler().fit_transform(final_df[['Reporting Economy GDP']])
    final_df['Scaled Partner Economy GDP'] = StandardScaler().fit_transform(final_df[['Partner Economy GDP']])
    final_df['Scaled Value'] = StandardScaler().fit_transform(final_df[['Value']])
    final_df['Scaled #trading partners'] = StandardScaler().fit_transform(final_df[['#trading partners']])
    final_df['Scaled import_duty'] = StandardScaler().fit_transform(final_df[['import_duty']])

    final_df = final_df[['Scaled Origin Output','scaled distance between origin and destination','Scaled Reporting Economy GDP','Scaled Value','Scaled Partner Economy GDP','Scaled #trading partners','FTA','Scaled import_duty']]
                    #,'#trading partners']]

    final_df['Client'] = data['Y Label']
    final_df=final_df.fillna(0)

    X = final_df.loc[:,final_df.columns != 'Client']
    Y = final_df['Client']

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

    model = Sequential()
    model.add(Dense(5000, input_dim=8, activation='tanh'))
    #odel.add(Dense(10000, activation='relu'))
    model.add(Dense(1000, activation='tanh'))
    model.add(Dense(500, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.01, momentum=0.9)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            ## Adding the callback for TensorBoard and History
    tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, Y_train,validation_data=(X_test,Y_test), callbacks=[tensorboard], epochs=10, batch_size=512)

    model.save('model.h5py')
    with file_io.FileIO('model.h5py', mode='rb') as input_f:
        with file_io.FileIO(job_dir + 'model.h5py', mode='w+') as output_f:
            output_f.write(input_f.read())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)  