import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import os


class Cluster(object):

    def __init__(self, path, plot=False):
        self.path = path
        self.plot = plot

        self.im = None
        self.imm_cl_RGB = None
        self.img_bw = None
        self.zoomed = None
        self.img_zm = None

        self.results = []

        self.import_file()

    def create_image(self, im_2D, Ncluster, labels, centroids):

        imm = im_2D.copy()

        # create the 'clustered'/quantized image [still in 2D]
        for k in range(Ncluster):
            ind = (labels == k)
            imm[ind, :] = centroids[k, :]
        return imm

    def import_file(self):

        filein = self.path

        # creating the cluster
        self.im = mpimg.imread(filein)

        # parsing from 3 dimension to 2.
        [N1, N2, N3] = self.im.shape
        im_2D = self.im.reshape((N1 * N2, N3))
        [Nr, Nc] = im_2D.shape
        Ncluster = 3
        kmeans = KMeans(n_clusters=Ncluster, random_state=0)
        kmeans.fit(im_2D)

        # getting the centroid and the N1*N2 classes/clusters each pixel belongs to
        centroids = kmeans.cluster_centers_.astype("uint8")
        labels = kmeans.labels_
        # create the 'clustered'/quantized image [still in 2D]
        imm = self.create_image(im_2D, Ncluster, labels, centroids)

        # Reshape to RGB
        self.imm_cl_RGB = imm.reshape((N1, N2, N3))

        se = np.sum((im_2D - imm) ** 2)
        print('Square distance = ' + str(se))

        img_tmp = (imm == np.min(centroids, axis=0)).reshape(N1, N2, N3)
        N1 = img_tmp.shape[0]
        N2 = img_tmp.shape[1]
        self.img_bw = np.zeros((N1, N2), dtype='uint8')

        for row in range(N1):
            for column in range(N2):
                if (img_tmp[row, column].all()):
                    self.img_bw[row, column] = 1

        # plot
        if self.plot:
            plt.figure(1)
            plt.imshow(self.im)
            plt.title("original image")
            plt.show()

            plt.figure(2)
            plt.imshow(self.imm_cl_RGB, interpolation=None)
            plt.title('Quantized image (3 colors, K-Means)')
            plt.show()

            plt.figure(3)
            plt.imshow(self.img_bw)
            plt.title("Black and White Image")
            plt.show()

        self.zoomed_image()

    def zoomed_image(self):

        allr = []
        allc = []
        for i in range(self.img_bw.shape[0] - 100):
            for j in range(self.img_bw.shape[1] - 100):
                if self.img_bw[i][j] == 1:
                    allr.append(i)
                    allc.append(j)

        row = allr[round(len(allr) / 2)]
        col = allc[round(len(allc) / 2)]

        print(row, col)

        #center = [round(self.img_bw.shape[0] / 2), round(self.img_bw.shape[1] / 2)]
        #print(center)
        center = [row, col]

        col_left_min_arr = []
        col_left_max_arr = []
        col_right_min_arr = []
        col_right_max_arr = []
        row_up_min_arr = []
        row_up_max_arr = []
        row_down_min_arr = []
        row_down_max_arr = []

        not_continuous = 0
        # Sopra al centro
        for r in range(center[0]):
            if not_continuous > 3:
                break
            iikc = self.img_bw[center[0] - r, :]
            if np.sum(iikc) > 5:
                ii = np.argwhere(iikc == 1)
                stop = len(ii)
                for x in range(stop - 1):
                    if ii[x] - ii[x + 1] < -10:
                        stop = x
                        break
                new_ii = ii[0:stop]
                if len(new_ii) > 0:
                    row_up_min_arr.append(new_ii.min())
                    row_up_max_arr.append(new_ii.max())
            else:
                not_continuous += 1

        not_continuous = 0
        # Sotto al centro
        for r in range(center[0]):
            if not_continuous > 3:
                break
            iikc = self.img_bw[center[0] + r, :]
            if np.sum(iikc) > 5:
                ii = np.argwhere(iikc == 1)
                stop = len(ii)
                for x in range(stop - 1):
                    if ii[x] - ii[x + 1] < -10:
                        stop = x
                        break
                new_ii = ii[0:stop]
                if len(new_ii) > 0:
                    row_down_min_arr.append(new_ii.min())
                    row_down_max_arr.append(new_ii.max())
            else:
                not_continuous += 1

        not_continuous = 0
        # Destra del centro
        for r in range(center[1]):
            if not_continuous > 3:
                break
            iikc = self.img_bw[:, center[1] + r]
            if np.sum(iikc) > 5:
                ii = np.argwhere(iikc == 1)
                stop = len(ii)
                for x in range(stop - 1):
                    if ii[x] - ii[x + 1] < -10:
                        stop = x
                        break
                new_ii = ii[0:stop]
                if len(new_ii) > 0:
                    col_right_min_arr.append(new_ii.min())
                    col_right_max_arr.append(new_ii.max())
            else:
                not_continuous += 1

        not_continuous = 0
        # Sinistra del centro
        for r in range(center[1]):
            if not_continuous > 3:
                break
            iikc = self.img_bw[:, center[1] - r]
            if np.sum(iikc) > 5:
                ii = np.argwhere(iikc == 1)
                stop = len(ii)
                for x in range(stop - 1):
                    if ii[x] - ii[x + 1] < -10:
                        stop = x
                        break
                new_ii = ii[0:stop]
                if len(new_ii) > 0:
                    col_left_min_arr.append(new_ii.min())
                    col_left_max_arr.append(new_ii.max())
            else:
                not_continuous += 1

        if np.amin(col_left_min_arr) < np.amin(col_right_min_arr):
            row_up = np.amin(col_left_min_arr)
        else:
            row_up = np.amin(col_right_min_arr)

        if np.amax(col_right_max_arr) > np.amax(col_left_max_arr):
            row_down = np.amax(col_right_max_arr)
        else:
            row_down = np.amax(col_left_max_arr)

        if np.amin(row_up_min_arr) < np.amin(row_down_min_arr):
            col_left = np.amin(row_up_min_arr)
        else:
            col_left = np.amin(row_down_min_arr)

        if np.amax(row_down_max_arr) > np.amax(row_up_max_arr):
            col_right = np.amax(row_down_max_arr)
        else:
            col_right = np.amax(row_up_max_arr)

        print('left: ', col_left)
        print('right: ', col_right)
        print('up: ', row_up)
        print('down: ', row_down)

        self.zoomed = self.img_bw[row_up:row_down, col_left:col_right]

        # plot
        if self.plot:
            plt.figure(4)
            plt.imshow(self.zoomed)
            plt.title("Zoomed Image")
            plt.show()

        self.zoomed_image_clear()

    def zoomed_image_clear(self):

        self.N1 = self.zoomed.shape[0]  # xlabel
        self.N2 = self.zoomed.shape[1]  # ylabel

        for x in range(self.N2):

            col = self.zoomed[:, x]
            one = np.argwhere(col == 1)
            if len(one) > 0:
                ones = np.concatenate(one, axis=0)

                for t in ones:
                    if x == 0 and t == 0:
                        tmp = self.zoomed[t:t + 2, x:x + 2]
                    elif x == 0 and t == self.N1:
                        tmp = self.zoomed[t - 1:t, x:x + 2]
                    elif x == self.N2 and t == 0:
                        tmp = self.zoomed[t:t + 2, x - 1:x]
                    elif x == self.N2 and t == self.N1:
                        tmp = self.zoomed[t - 1:t, x - 1:x]
                    elif x == 0:
                        tmp = self.zoomed[t - 1:t + 2, x:x + 2]
                    elif x == self.N2:
                        tmp = self.zoomed[t - 1:t + 2, x - 1:x]
                    elif t == 0:
                        tmp = self.zoomed[t:t + 2, x - 1:x + 2]
                    elif t == self.N1:
                        tmp = self.zoomed[t - 1:t, x - 1:x + 2]
                    else:
                        tmp = self.zoomed[t - 1:t + 2, x - 1:x + 2]

                    if not np.sum(tmp) > 3:
                        self.zoomed[t, x] = 0
                    else:
                        continue
        #plot
        if self.plot:
            plt.figure(5)
            plt.imshow(self.zoomed)
            plt.title("Zoomed Image - Clear")
            plt.show()

        self.contour()

    def contour(self):

        self.img_zm = np.zeros((self.N1, self.N2), dtype='uint8')

        for r in range(self.N2):
            iikc = self.zoomed[:, r]
            if np.sum(iikc) > 20:
                ii = np.argwhere(iikc == 1)
                iimin = ii.min()
                iimax = ii.max()
                self.img_zm[iimin, r] = 1
                self.img_zm[iimax, r] = 1

        for r in range(self.N1):
            iikc = self.zoomed[r, :]
            if np.sum(iikc) > 20:
                ii = np.argwhere(iikc == 1)
                iimin = ii.min()
                iimax = ii.max()
                self.img_zm[r, iimin] = 1
                self.img_zm[r, iimax] = 1

        self.img_zm[0, :] = 0
        self.img_zm[-1, :] = 0
        self.img_zm[:, 0] = 0
        self.img_zm[:, -1] = 0

        #plot
        if self.plot:
            plt.figure(6)
            plt.imshow(self.img_zm)
            plt.title("Contour Image")
            plt.show()

        self.optimazed_contour()

    def optimazed_contour(self):

        for i in range(self.zoomed.shape[0]):
            for j in range(self.zoomed.shape[1]):
                if self.zoomed[i, j] == 1:
                    if (j - 2) > 0 and (j + 2) < self.zoomed.shape[1] and (
                            (self.zoomed[i, j + 1] != 1 and self.zoomed[i, j + 2] != 1) or (
                            self.zoomed[i, j - 1] != 1 and self.zoomed[i, j - 2] != 1)):
                        self.img_zm[i, j] = 1

        for j in range(self.zoomed.shape[1]):
            for i in range(self.zoomed.shape[0]):
                if self.zoomed[i, j] == 1:
                    if (i - 2) > 0 and (i + 2) < self.zoomed.shape[0] and (
                            (self.zoomed[i + 1, j] != 1 and self.zoomed[i + 2, j] != 1) or (
                            self.zoomed[i - 1, j] != 1 and self.zoomed[i - 2, j] != 1)):
                        self.img_zm[i, j] = 1

        #plot
        if self.plot:
            plt.figure(7)
            plt.imshow(self.img_zm)
            plt.title("Optimazed Contour - version 1")
            plt.show()

        for x in range(self.N2):

            col = self.img_zm[:, x]
            one = np.argwhere(col == 1)
            if len(one) > 0:
                ones = np.concatenate(one, axis=0)
                for t in ones:
                    if x == 0 and t == 0:
                        tmp = self.img_zm[t:t + 2, x:x + 2]
                    elif x == 0 and t == self.N1:
                        tmp = self.img_zm[t - 1:t, x:x + 2]
                    elif x == self.N2 and t == 0:
                        tmp = self.img_zm[t:t + 2, x - 1:x]
                    elif x == self.N2 and t == self.N1:
                        tmp = self.img_zm[t - 1:t, x - 1:x]
                    elif x == 0:
                        tmp = self.img_zm[t - 1:t + 2, x:x + 2]
                    elif x == self.N2:
                        tmp = self.img_zm[t - 1:t + 2, x - 1:x]
                    elif t == 0:
                        tmp = self.img_zm[t:t + 2, x - 1:x + 2]
                    elif t == self.N1:
                        tmp = self.img_zm[t - 1:t, x - 1:x + 2]
                    else:
                        tmp = self.img_zm[t - 1:t + 2, x - 1:x + 2]

                    if not np.sum(tmp) > 1:
                        self.img_zm[t, x] = 0

        #plot
        if self.plot:
            plt.figure(8)
            plt.imshow(self.img_zm)
            plt.title("Contour Optimazed - version 2")
            plt.show()

        self.ratio()

    def ratio(self):

        from collections import Counter

        perimeter = Counter(self.img_zm.reshape(self.img_zm.shape[0] * self.img_zm.shape[1]))[1]
        area = Counter(self.zoomed.reshape(self.zoomed.shape[0] * self.zoomed.shape[1]))[1]
        perimeter_circle = np.sqrt(float(area) / np.pi) * 2 * np.pi
        print(area, perimeter, perimeter_circle, float(perimeter) / perimeter_circle)

        name = self.path.split(".")[0]

        json = {
            'file': name,
            'perimeter':perimeter,
            'perimeter_circle': perimeter_circle,
            'ratio': float(perimeter) / perimeter_circle
        }

        self.results.append(json)
        return self.results


if __name__ == "__main__":

    iteration = True

    if iteration:
        directory = os.fsencode('images')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)

            # check if it's an jpg file
            if filename.endswith(".jpg"):
                print(filename.split(".")[0])
                cluster = Cluster('images/'+filename, False)
            else:
                raise ValueError('No correct file format for {}'.format(filename))
    else:
       cluster = Cluster('images/low_risk_1.jpg', True)