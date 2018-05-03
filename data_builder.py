import os
import cv2
import sys
import numpy as np
import scipy.io as io
from collections import Counter


def create_training_data(in_dir, out_dir, patch_size, stride):
    patch_count, cell_count = 0, 0
    b_size = patch_size // 2
    start = int(sorted(os.listdir(in_dir))[0].replace("img", ""))

    patch_files, classifications = [], []
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for img_num in range(start, start + len(os.listdir(in_dir))):
        image = cv2.imread(in_dir + "img" + str(img_num) + "/img" + str(img_num) + ".bmp")
        border_image = cv2.copyMakeBorder(image, b_size, b_size, b_size, b_size, cv2.BORDER_WRAP)

        epi = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_epithelial.mat")["detection"]
        fib = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_fibroblast.mat")["detection"]
        inf = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_inflammatory.mat")["detection"]
        other = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_others.mat")["detection"]
        point_sets = [epi, fib, inf, other]

        for p in range(len(point_sets)):
            for point in point_sets[p]:
                patches = []
                for i in range(int(point[0]) - 3, int(point[0]) + 4, stride):
                    for j in range(int(point[1]) - 3, int(point[1]) + 4, stride):
                        patches.append(border_image[j - b_size:j + b_size + 1, i - b_size:i + b_size + 1])
                        for x in [0, 1, 2]:
                            patches.append(cv2.flip(patches[0], x))
                            for y in [1, 2, 3]:
                                patches.append(np.rot90(patches[0], y))
                        for m in range(len(patches)):
                            cv2.imwrite(out_dir + '/' + str(patch_count) + ".bmp", patches[m])
                            if patch_count % 100 == 0:
                                print(str(patch_count) + " Completed!")
                            patch_files.append(str(patch_count) + ".bmp")
                            classifications.append(p)
                            patch_count += 1
                            cell_count += 1
                        patches = []
                print("Patches per cell: " + str(cell_count))
                cell_count = 0
        print("Image " + str(img_num) + " Completed!")
    np.save(out_dir + "/values.npy", np.array([patch_files, classifications]))
    print("Done!")
    print("Number of Patches: " + str(patch_count))
    print("Patch Types: " + str(Counter(classifications)))


if __name__ == "__main__":
    in_dir = sys.argv[1]
    if in_dir[-1] != '/':
        in_dir += '/'

    out_dir = sys.argv[2]
    if out_dir[-1] != '/':
        out_dir += '/'

    create_training_data(in_dir, out_dir, int(sys.argv[3]), int(sys.argv[4]))
