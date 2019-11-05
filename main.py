import hyperNet, embedder
import torch


if __name__ == '__main__':

    embedder = embedder.DummyEmbedder()
    dataset_parmas = {"train": None, "dev": None, "test": None}
    hypernet = hyperNet.HypetNet(embedder, dim = 768, dropout = 0.1, effective_rank = 64, dataset_parmas = dataset_parmas)
    optimizer = torch.optim.Adam(hypernet.parameters())
    text = [ ["Hello", "how", "are", "you", "today"], ["yes", "why", "not", "?", "?"] , ["yes", "why", "not", "?", "?"]]
    
    result = hypernet.training_step(text,0)
    #loss = result["loss"]
    #loss.backward()
    #optimizer.step()
    print(result)
    print("The model has {} million parameters".format(sum([param.nelement() for param in hypernet.parameters()])/1e6))
