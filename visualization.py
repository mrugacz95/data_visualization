import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, manifold

from mds import mds_custom, mds_sklearn


def isomap(data, k):
    embedding = manifold.Isomap(n_components=2, n_neighbors=k)
    return embedding.fit_transform(data)


def tsne(data):
    embedding = manifold.TSNE(n_components=2)
    return embedding.fit_transform(data)


def lle(data):
    embedding = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=7)
    return embedding.fit_transform(data)


def plot_data(data, title, labels=None, colors=None):
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1], c=colors)
    if labels is not None:
        for i, txt in enumerate(labels):
            plt.text(data[i, 0], data[i, 1], txt, va="center")


def make_experiment(data, name=None, labels=None, figsize=(20, 10), colors=None):
    plt.figure(figsize=figsize)
    plt.subplot(231)
    Y = mds_custom(data)
    plot_data(Y, 'MDS classic (pca)', labels, colors)
    plt.subplot(232)
    Y = mds_sklearn(data)
    plot_data(Y, 'MDS sklearn (smacof)', labels, colors)
    plt.subplot(233)
    Y = isomap(data, 3)
    plot_data(Y, 'Isomap (k=3)', labels, colors)
    plt.subplot(234)
    Y = isomap(data, 5)
    plot_data(Y, 'Isomap (k=5)', labels, colors)
    plt.subplot(235)
    Y = tsne(data)
    plot_data(Y, 'Tsne', labels, colors)
    plt.subplot(236)
    Y = lle(data)
    plt.suptitle(name)
    plot_data(Y, 'lle', labels, colors)
    plt.savefig(f'plots/{name}.jpg')


def main():
    # cars
    data = pd.read_csv('data/cars.csv', index_col=False, quotechar="'", header=None)
    labels = data[0]
    data = data.values[:, 1:]
    make_experiment(data, 'Cars', labels)
    # swissroll
    data, t = datasets.make_swiss_roll()
    make_experiment(data, "Swissroll", colors=t)
    # pokemon
    data = pd.read_csv('data/pokemon.csv', delimiter=',', index_col=0)
    labels = data[data['Generation'] == 1]['Name']
    data = data.drop('Name', axis=1)
    strengths = pd.read_csv('data/strengths.csv', index_col=0)
    data['Type 2'] = data['Type 2'].fillna(data['Type 1'])  # double pokemon type if only one is present
    data = data.join(strengths, on='Type 1', rsuffix="_t1")
    data = data.join(strengths, on='Type 2', rsuffix="_t2")
    data = data.drop(columns=['Type 1', 'Type 2'])

    data = data[data['Generation'] == 1]
    data['Legendary'] = data['Legendary'].apply(lambda x: 1 if x else 0)
    make_experiment(data, "Pokemon", labels, figsize=(40, 30))


if __name__ == '__main__':
    main()
