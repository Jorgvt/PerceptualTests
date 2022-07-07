class NaiveModel():
    """
    Model that does absolutelly nothing.
    Can be used as a first baseline to compare with
    more complicated models.
    """
    def __call__(self, X):
        return X

    def predict(self, X):
        return self(X)