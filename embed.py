import networkx as nx
import numpy as np
from node2vec.node2vec import Node2Vec

nx_graph = nx.Graph()

class EmbedGraph():
    def __init__(self, adj_matrix: np.ndarray, normalize: bool,
                n2v_dimensions = 10, n2v_walks = 200, n2v_workers = 4):
        self.adj_matrix = adj_matrix
        self.dimensions = n2v_dimensions
        self.walks = n2v_walks
        self.workers = n2v_workers
        
        self.model = None
        
        if normalize:
            adj_min = self.adj_matrix.min()
            adj_max = self.adj_matrix.max()
            
            self.adj_matrix = (self.adj_matrix - adj_min) / (adj_max-adj_min)

    
        edges = list()
        weights = list()
        
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix)):
                if self.adj_matrix[i,j] != 0:
                    edges.append((i+1, j+1))
                    weights.append((i+1, j+1, self.adj_matrix[i,j]))
    
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(weights)
        
        self.n2v = Node2Vec(self.G,
                            dimensions=self.dimensions,
                            num_walks=self.walks,
                            workers=self.workers)
        
    def fit_n2v(self, window_size = 64, min_count = 2, batch_words = 4):
        self.model = self.n2v.fit(window=window_size,
                        min_count=min_count,
                        batch_words=batch_words)
        
    @property
    def embeddings(self):
        if self.model is None:
            self.fit_n2v()
    
        return self.model.wv.vectors