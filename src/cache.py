
class Cache:
    def __init__(self, n_layers):
        self.embeddings = {i: [] for i in range(1,n_layers+1)}

    @property
    def size(self):
        return len(self.embeddings)

    def get(self, i):
        assert i < len(self.embeddings)
        return self.embeddings[i]

    def store(self, x, i):
        self.embeddings[i].append(x)
