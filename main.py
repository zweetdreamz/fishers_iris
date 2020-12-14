import random
import math
import matplotlib.pyplot as plt


def encircle(x, y, ax=None, **kw):
    from scipy.spatial import ConvexHull
    import numpy as np
    if not ax: ax = plt.gca()
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)
    ax.add_patch(poly)


def read_data(path):
    with open(path, 'r', encoding='windows-1251') as file:
        content = file.read().splitlines()
        data = list()
        for i in content:
            data.append(list(map(lambda x: x if 'Iris' in x else float(x), i.split(','))))
        return data


def euclidean(p1, p2):
    return math.sqrt(sum([(i_1 - i_2) ** 2 for i_1, i_2 in zip(p1, p2)]))


def means(array):
    if len(array) > 0:
        tmp = [0] * len(array[0])
        for i in array:
            for j in range(len(array[0])):
                tmp[j] += i[j]

        return [i / len(array) for i in tmp]
    else:
        return [random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 10), random.uniform(0, 10)]


data = read_data('iris.csv')
data = random.sample(data, len(data))

split = (0.8 * data.__len__()).__int__()
train = data[:split]
test = data[split:]

k = 3
steps = 0

prev_centroids = [[random.uniform(0, 10),
                   random.uniform(0, 10),
                   random.uniform(0, 10),
                   random.uniform(0, 10)] for _ in range(k)]
centroids = [[random.uniform(0, 10),
              random.uniform(0, 10),
              random.uniform(0, 10),
              random.uniform(0, 10)] for _ in range(k)]
print('Start centroids: ', *centroids)

while abs(sum([sum(i) for i in prev_centroids]) - sum([sum(i) for i in centroids])) > 0.0001:
    current_clusters = {i: list() for i in range(k)}
    for i in train:
        point = i[:-1]
        distances = [[centroid_num, euclidean(point, centroid)] for centroid_num, centroid in
                     enumerate(centroids)]
        distances.sort(key=lambda x: x[1])
        if point not in current_clusters[distances[0][0]]:
            current_clusters[distances[0][0]].append(i)

    prev_centroids = centroids
    centroids = [means([j[:-1] for j in i]) for i in current_clusters.values()]
    steps += 1

print('Target centroids: ', *centroids)
print('Steps: ', steps)

my_cluster = [[i, current_clusters[i],
               max(set([j[-1] for j in current_clusters[i]]), key=[j[-1] for j in current_clusters[i]].count)] for i in
              current_clusters]
print('Cluster:')
for i in my_cluster:
    print(i[2], len(i[1]))

count = 0
for i in test:
    distances = [[centroid_num, euclidean(i[:-1], centroid)] for centroid_num, centroid in
                 enumerate(centroids)]
    distances.sort(key=lambda x: x[1])
    if i[-1] == my_cluster[distances[0][0]][2]:
        count += 1

print('Accuracy: ', count / len(test))

colors = {'Iris-versicolor': 'purple', 'Iris-virginica': 'red', 'Iris-setosa': 'green'}
plt.gca().set(xlim=(3.5, 8), ylim=(1, 5.5), xlabel='Длина чашелистника', ylabel='Ширина чашелистника')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.grid()

for i in my_cluster[0][1]:
    plt.scatter(i[0], i[1], s=12, color=colors[my_cluster[0][2]], label=my_cluster[0][2])

for i in my_cluster[1][1]:
    plt.scatter(i[0], i[1], s=12, color=colors[my_cluster[1][2]], label=my_cluster[1][2])

for i in my_cluster[2][1]:
    plt.scatter(i[0], i[1], s=12, color=colors[my_cluster[2][2]], label=my_cluster[2][2])

encircle([i[0] for i in my_cluster[0][1]], [i[1] for i in my_cluster[0][1]], ec="k", fc="gold", alpha=0.2, linewidth=0)
encircle([i[0] for i in my_cluster[1][1]], [i[1] for i in my_cluster[1][1]], ec="k", fc="tab:blue", alpha=0.2,
         linewidth=0)
encircle([i[0] for i in my_cluster[2][1]], [i[1] for i in my_cluster[2][1]], ec="k", fc="tab:orange", alpha=0.2,
         linewidth=0)
plt.savefig("iris.png", dpi=400)
plt.show()

p = 1
