import numpy as np


class RepresentationModel:
    def set_train(self, train):
        raise NotImplementedError('A representation model must have a set train function')

    @staticmethod
    def _get_vec(d2v, key):
        """Return the vector for *key* from *d2v*, or a zero vector if missing.

        Infers the required dimension from the first existing entry in *d2v*.
        Handles both plain dicts (KeyError) and defaultdicts-with-empty-list.
        """
        try:
            vec = d2v[key]
        except (KeyError, TypeError):
            vec = None

        # Empty defaultdict(list) hit or key not found → fallback
        if vec is None or (isinstance(vec, list) and len(vec) == 0):
            dim = None
            for v in d2v.values():
                if isinstance(v, np.ndarray):
                    dim = v.shape[0]
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    dim = len(v)
                else:
                    continue
                break
            return np.zeros(dim if dim is not None else 0)

        return np.asarray(vec)

    def vectorize_interaction(self, interaction):
        return np.hstack([self._get_vec(self.ligand2vec, interaction['ligand_id']),
                          self._get_vec(self.prot2vec, interaction['prot_id'])])
