import sys
from onnx import onnx_pb
from onnx_coreml import convert

INV_CLASS = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}
CLASSES = list(INV_CLASS.values())

model_in = sys.argv[1]
model_out = sys.argv[2]

model_file = open(model_in, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
coreml_model = convert(model_proto, mode='classifier', class_labels=CLASSES) #image_input_names=['0'], image_output_names=['186'])
coreml_model.save(model_out)

'''def convert(model,
            mode='classifier,
            image_input_names=[],
            preprocessing_args={},
            image_output_names=[],
            deprocessing_args={},
            class_labels=None,
            predicted_feature_name='classLabel',
            add_custom_layers = False,
            custom_conversion_functions = {})'''


#import coremltools
#from coremltools.models import MLModel

#spec = coremlmodel.get_spec()
#new_mlmodel = MLModel(spec)
#coremltools.utils.rename_feature(spec, 'old_output_name', 'new_output_name')
#coremltools.utils.save_spec(spec, 'model_new_output_name.mlmodel')