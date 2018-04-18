import os
import cv2
import sys
import numpy as np
import scipy.io as io
from collections import Counter
from scipy.spatial import distance


def create_training_data(in_dir, out_dir, patch_size):
    patch_count = 0
    border_size = patch_size // 2
    start = int(sorted(os.listdir(in_dir))[0].replace("img", ""))

    patch_files, classifications = [], []
    
    for img_num in range(start, start + len(os.listdir(in_dir))):
        image = cv2.imread(in_dir + "img" + str(img_num) + "/img" + str(img_num) + ".bmp")
        border_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_WRAP)

        epi = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_epithelial.mat")["detection"]
        fib = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_fibroblast.mat")["detection"]
        inf = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_inflammatory.mat")["detection"]
        other = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_others.mat")["detection"]
        points = [epi, fib, inf, other]

        for pi in range(border_size, image.shape[0] + border_size, patch_size):
            for pj in range(border_size, image.shape[1] + border_size, patch_size):
                min_dist = 1000
                label = -1
                for i in range(len(points)):
                    for point in points[i]:
                        dist = distance.euclidean((point[0], point[1]), (pi - border_size, pj - border_size))
                        if dist < min_dist:
                            min_dist = dist
                            label = i
                if min_dist >= 6:
                    continue
                patch = border_image[pj - border_size:pj + border_size + 1, pi - border_size:pi + border_size + 1]
                patches = [patch]
                for i in [0, 1, 2]:
                    patches.append(cv2.flip(patch, i))
                    for j in [1, 2, 3]:
                        patches.append(np.rot90(patch, j))
                for p in range(len(patches)):
                    cv2.imwrite(out_dir + '/' + str(patch_count) + ".bmp", patches[p])
                    if patch_count % 100 == 0:
                        print(str(patch_count) + " Completed!")
                    patch_files.append(str(patch_count) + ".bmp")
                    classifications.append(label)
                    patch_count += 1
        print("Image " + str(img_num) + " Completed!")
    np.save(out_dir + "/values.npy", np.array([patch_files, classifications]))
    print("Done!")
    print("Number of Patches: " + str(patch_count))
    print("Patch Types: " + str(Counter(classifications)))

    
if __name__ == "__main__":
    in_dir = sys.argv[2]
    if in_dir[-1] != '/':
        in_dir += '/'

    out_dir = sys.argv[3]
    if out_dir[-1] != '/':
        out_dir += '/'

    if sys.argv[1].lower() == 'train':
        create_training_data(in_dir, out_dir, int(sys.argv[4]))
