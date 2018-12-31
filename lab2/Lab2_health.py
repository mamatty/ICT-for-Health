import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import subprocess as sp
import random
import json
from pprint import pprint


class Kidney(object):

    def __init__(self, conf, drop_nan=False):

        with open(conf) as f:
            self.data_json = json.load(f)
        self.path = self.data_json['path']
        self.feat_names = self.data_json['feat_names']
        self.stripped = self.data_json['stripped']
        self.drop_nan = drop_nan

        self.clf = None
        self.file_name = None
        self.fig_name = None

    def __import_data(self):
        # check the validity of the input file
        try:
            # creating the DataFrame
            ckd = pd.read_csv(self.path, sep=',', skiprows=29, header=None, na_values=['', ' ','?', '\t?'],
                              skipinitialspace=True, names=self.feat_names)

            for col in self.stripped:
                ckd[col] = ckd[col].str.strip()

            if not self.drop_nan:
                try:
                    ckd = ckd.fillna(self.data_json['fill_na'])
                except:
                    raise ValueError('Value {} already present inside the Dataset! Change value for NaN value substitution'.format(self.data_json['fill_na']))
            else:
                ckd = ckd.dropna(axis=0)

            # pre-processing the dataset - creating a not labeled version
            for val in self.data_json['substitutions']:
                ckd = ckd.replace(val['str'], val['value'])

            return ckd

        # raise exception if there is something wrong
        except:
            raise IOError("Error: File does not appear to exist.")

    def export(self):

        tt = self.__import_data()

        if not self.drop_nan:
            self.file_name = 'ckd_nan_replaced'
            self.fig_name = 'Tree_NaN_replaced'
        else:
            self.file_name = 'ckd_nan_dropped'
            self.fig_name = 'Tree_NaN_dropped'

        writer = pd.ExcelWriter('{}.xlsx'.format(self.file_name))
        tt.to_excel(writer, 'Sheet1')
        writer.save()

    def Decision_tree(self):

        random.seed(1000)
        np.random.seed(50)

        ckd = self.__import_data()
        data = ckd.drop(labels='class', axis=1)
        target = ckd['class']
        self.clf = tree.DecisionTreeClassifier("entropy", max_features=1).fit(data, target)
        classes = ['notckd', 'ckd']
        dot_data = tree.export_graphviz(self.clf, out_file='{}.dot'.format(self.fig_name), feature_names=self.feat_names[0:24],
                                        class_names=classes, filled=True, rounded=True, special_characters=True)

        self.__grafication()

    def __grafication(self):

        input_name = '{}.dot'.format(self.fig_name)
        output_name = '{}.png'.format(self.fig_name)
        png = sp.run(['dot', '-Tpng', input_name, '-o', output_name], stdout=sp.PIPE, shell=True)
        print(png.stdout.decode('utf-8'))

        import matplotlib.pyplot as plt
        import matplotlib.image as mplimg

        img = mplimg.imread('{}.png'.format(self.fig_name))
        plt.figure(figsize=(45, 15))
        imgplot = plt.imshow(img)
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.stem(self.clf.feature_importances_)

        if not self.drop_nan:
            title = 'Feature Importances (Nan Replaced)'
        else:
            title = 'Feature Importances (Nan Dropped)'

        plt.title(title, fontsize=15)
        plt.xticks(range(0, 25), self.feat_names[0:24], rotation=90, fontsize=12)
        plt.savefig('{}.png'.format(title))
        plt.show()

if __name__ == '__main__':
    CONF = 'conf.json'
    kidney = Kidney(CONF)
    kidney.export()
    kidney.Decision_tree()
