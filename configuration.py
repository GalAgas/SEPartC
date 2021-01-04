class ConfigClass:
    def __init__(self):
        # link to a zip file in google drive with your pretrained model
        self._model_url = None
        # False/True flag indicating whether the testing system will download
        # and overwrite the existing model files. In other words, keep this as
        # False until you update the model, submit with True to download
        # the updated model (with a valid model_url), then turn back to False
        # in subsequent submissions to avoid the slow downloading of the large
        # model file with every submission.
        self._download_model = False

        self.corpusPath = ''
        self.savedFileMainFolder = ''
        self.saveFilesWithStem = self.savedFileMainFolder + "/WithStem"
        self.saveFilesWithoutStem = self.savedFileMainFolder + "/WithoutStem"
        self.toStem = False

        print('Project was created successfully..')

    def get__corpusPath(self):
        return self.corpusPath

    def set_corpusPath(self, path):
        self.corpusPath = path

    def get_toStem(self):
        return self.toStem

    def set_toStem(self, to_stem):
        self.toStem = to_stem

    def get_savedFileMainFolder(self):
        return self.savedFileMainFolder

    def set_savedFileMainFolder(self, path):
        self.savedFileMainFolder = path

    def get_model_url(self):
        return self._model_url

    def get_download_model(self):
        return self._download_model