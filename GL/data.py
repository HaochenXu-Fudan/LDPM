"""Small data containers shared by the Group Lasso drivers."""


class Data:
    def __init__(self):
        self.X_train = None
        self.X_validate = None
        self.X_test = None
        self.y_train = None
        self.y_validate = None
        self.y_test = None


class Data_with_Info:
    def __init__(self, data, settings, data_index=0):
        self.data = data
        self.settings = settings
        self.data_index = data_index
