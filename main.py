import hyperNet, embedder



if __name__ == '__main__':

    embedder = embedder.DummyEmbedder()
    dataset_parmas = {"train": None, "dev": None, "test": None}
    hypernet = hyperNet.HypetNet(embedder, dim = 768, dropout = 0.1, effective_rank = 128, dataset_parmas = dataset_parmas)

