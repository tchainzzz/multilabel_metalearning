from load_data_tf import BigEarthNetDataset
import numpy as np

from tqdm import tqdm

if __name__ == '__main__':
    dataset = BigEarthNetDataset("../SmallEarthNet")
    print("Calculating channel means and standard deviation:")
    sums = np.zeros(3)
    sum_sqs = np.zeros(3)
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        w, h, _ = img.shape
        sums += img.sum(axis=(0, 1))
        sum_sqs += np.square(img).sum(axis=(0, 1))
    n = len(dataset) * w * h
    means = sums / n
    print(sums, sum_sqs)
    stds = np.sqrt(n * sum_sqs - np.square(sums)) / n
    print(means, stds)

