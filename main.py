from LoadData import LoadData
from FoolboxModel import ResNet50
from SparseSimBA import SparseSimBA
from AttackEngine import AttackEngine
from numpy import shape

## Load Data
loadData = LoadData("/Users/gasimazhari/fyp/data/val.txt", "/Users/gasimazhari/fyp/data/ILSVRC2012_img_val/")
model = ResNet50()
if (model.metadata()[3]):
    x, y = loadData.get_data_bgr([(0, 0)], model.metadata()[0], model.metadata()[1])
else:
    x, y = loadData.get_data([(0, 0)], model.metadata()[0], model.metadata()[1])

attack_engine = AttackEngine(x, y, model, [SparseSimBA])

print(shape(x))
print(y)
