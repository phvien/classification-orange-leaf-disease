import logging
from trainers import TrainingHelper
from predictors import PredictHelper


logging.basicConfig(filename='predicters.log', level=logging.INFO)
logging.info('='*100)


# runner = TrainingHelper(svm_model=True)


paths = 'dataset/demo01'
classes= ['Ghẻ Nhám', 'Lá Khỏe', 'Rầy Phấn Trắng', 'Vàng Lá Gân Xanh', 'Váng Lá Thối Rễ']
model = 'model/leaf_disease_detection.model'

runner = PredictHelper(paths, classes, model)

