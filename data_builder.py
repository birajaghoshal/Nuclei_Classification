import os
import cv2
import sys
import numpy as np
import scipy.io as io
from collections import Counter


def data_builder(in_dir, out_dir, patch_size, stride):
    patch_count, cell_count = 0, 0
    start = int(sorted(os.listdir(in_dir))[0].replace("img", ""))
    b = patch_size // 2

    patch_files, classifications = [], []
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for img_num in range(start, start + len(os.listdir(in_dir))):
        image = cv2.imread(in_dir + "img" + str(img_num) + "/img" + str(img_num) + ".bmp")
        border_image = cv2.copyMakeBorder(image, patch_size, patch_size, patch_size, patch_size, cv2.BORDER_WRAP)
        epi = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_epithelial.mat")["detection"]
        fib = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_fibroblast.mat")["detection"]
        inf = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_inflammatory.mat")["detection"]
        other = io.loadmat(in_dir + "img" + str(img_num) + "/img" + str(img_num) + "_others.mat")["detection"]
        point_sets = [epi, fib, inf, other]

        for p in range(len(point_sets)):
            for point in np.round(point_sets[p]).astype(int):
                patches = []
                for i in range(round(point[0]) - 3 + patch_size, round(point[0]) + 3 + patch_size, stride):
                    for j in range(round(point[1]) - 3 + patch_size, round(point[1]) + 3 + patch_size, stride):
                        patch = border_image[j - b: j + b + 1, i - b: i + b + 1]
                        patches.append(patch)
                        for x in [0, 1, 2]:
                            patch = cv2.flip(patches[0], x)
                            patches.append(patch)
                            for y in [1, 2, 3]:
                                temp_patch = np.rot90(patch, y)
                                patches.append(temp_patch)
                        for m in range(len(patches)):
                            cv2.imwrite(out_dir + '/' + str(patch_count) + ".bmp", patches[m])
                            patch_count += 1
                            if patch_count % 1000 == 0:
                                print(str(patch_count) + " Completed!")
                            patch_files.append(str(patch_count) + ".bmp")
                            classifications.append(p)
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

    data_builder(in_dir, out_dir, int(sys.argv[3]), int(sys.argv[4]))
