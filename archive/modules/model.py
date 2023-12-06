# CNN backbone model to implement the R-CNN
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense
from keras import Model

class Backbone:
    def __init__(self, base_model: str = "vgg16"):
        base_model = base_model.lower()
        if base_model not in ["vgg16", "resnet50"]:
            raise Exception(f"{base_model} model is not currently supported.")
        elif base_model == "vgg16":
            self._base_model = VGG16(weights = "imagenet", include_top = True)
            self._base_model_name = "VGG16"
        else:
            self._base_model = ResNet50(weights = "imagenet", include_top = True)
            self._base_model_name = "ResNet50"

        # Disable training of layers
        for layer in self._base_model.layers:
            layer.trainable = False

        features = self._base_model.layers[-2].output
        output_layer = Dense(units = 2, activation = "softmax")(features)
        self._base_model = Model(inputs = self._base_model.inputs, 
                            outputs = output_layer, name = "RCNN-" + self._base_model_name)

        tensor = self._base_model.inputs[0]
        self._input_shape = tuple(tensor.shape[1:])   # (224, 224, 3) in case of VGG16

    def get_model_config(self):
        return {"model_name": self._base_model_name,
                "input_shape": self._input_shape,
                "output_shape": (2,) }

    def summary(self):
        return self._base_model.summary()     

