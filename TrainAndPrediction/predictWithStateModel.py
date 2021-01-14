import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageOps
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T

path = '/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/'

datasetSize= 0
class PennFudanDataset(torch.utils.data.Dataset): 
    # Dataseti oluşturan onu modifiye eden sınıf
    def __init__(self, root, transforms=None): # yapıcı method
        global datasetSize
        self.root = root# root : dataset için resimlerin yolu 
        self.transforms = transforms #Bu transforms.py dosyasından çektiğimiz Compose sınıfı, amacı dataseti modifiye etmek.

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images")))) #bütün orjinal resimlerin isim listesi
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks")))) # bütün maskelerin isim listesi
        """
        #Burada Elimde olan Gpu ile alabildiğim maksimum dataset boyutu 444 orjinal resim olduğu için geriye kalanları listeden siliyorum.
        theValue = 200
        del self.imgs[theValue:] 
        del self.masks[theValue:] 
        """
        datasetSize = len(self.imgs)
    def __getitem__(self, idx): #Bu method pennfudandataset sınıfına ait bir nesnenin döndürülmesi halinde çağırılan özel bir methoddur.
        #  imageleri ve maskeleri değişkenlere atar
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        #idx değişkeni veriseti dönerken kendiliğinden artan indextir.Yani veriseti listesinin boyutu kadar döner.

        img = Image.open(img_path).convert("RGB") # Image operatörü PIL tipinde bir resime erişmek içindir.Burada-
        #yolunu verdiğimiz her tek resim için img değişkenine bir tane resim atanır ve bu resmi rgb formatına çeviririz.
        mask = Image.open(mask_path)#aynı şeyi maskeler için yapıyoruz fakat!, maskeleri rgb ye çevirmiyoruz çünkü her ayrı renk-
        #pikseli ayrı bir blob'a denk gelecek şekilde ayarlandığı için greyscale kalmalı!

        mask = np.array(mask)
        # maskeyi numpy array şekline çeviriyoruz

        obj_ids = np.unique(mask) # burada dizide olan eleman çeşidini görmek için np.uniqiue methodunu kullanıyoruz.
        #yani bunun içeriği 3 tane blob var ise  arkaplan dahil olmak üzere 4 tane eleman olacak ([0,1,2,3]).
        # aynı zamanda np.unique methodu verileri küçükten büyüğe sıralar.

        
        obj_ids = obj_ids[1:] # ilk eleman arkaplan olduğundan onu siliyoruz.

        masks = mask == obj_ids[:, None, None]#blobların olduğu pikselleri True , olmadığı pikselleri false yaparak masks numpy dizisini ayarlıyoruz


        # her maskenin bounding box koordinatlarını alma aşaması : 
        num_objs = len(obj_ids) # blob sayısını num_objs değişkenine setler.
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])# masks dizisindeki true olan(yani blobun olduğu x ve y koordinatı) x dizisi ve y dizisi olarak döndürür-
            #ve bu iki diziyi pos'a setler
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            #x ve y koordinatlarının max ve min değerlerini değişkenlere atar.-
            # Bu değişkenler bizim bounding box dikdörtgenin köşegen koordinatlarıdır.
            boxes.append([xmin, ymin, xmax, ymax])#sonunda boxes'dizisine eleman olarak verilir.

        boxes = torch.as_tensor(boxes, dtype=torch.float32)#boxes dizisini torch.float32 tipinde bir tensor'a çevirdik
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64) #blob sayısını labels adındaki diziye atıyoruz.-
        #Bu dizide blob sayısı kadar eleman olsa ve bunların hepsi 1 olsa yeterlidir.Çünkü sadece 1 etiketimiz var.
        masks = torch.as_tensor(masks, dtype=torch.uint8) #maskelerimizide tensor'a çeviriyoruz.

        image_id = torch.tensor([idx])# idx değerini bir nevi image_id olarak tutuyoruz(amaç resmin idx'ine erişebilme)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # bounding box'un alanını buluyoruz.

        #print('ALAN ================== ' ,area)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)#blob sayısı kadar 0 matrisi oluşturur

        #şimdi tüm bu özelliklerimizi target adında bir listeye atalım.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        #Bu if blogunda dataset'deki veriler modifiye edilecekse yani bir transforms değişkenimiz içerisinde objemiz var ise
        if self.transforms is not None:
            img, target = self.transforms(img, target)#img ve target'i modifiye et

        return img, target

    def __len__(self):
        return len(self.imgs)#veri setindeki resimlerin sayısını döndürür.
   
def get_instance_segmentation_model(num_classes):# pretrained modeli döndüren method
    #num_classes'ı aşağıda 2 vereceğiz-
    #2 vermemizin sebebi 1 etiketimizin bloblar diğer etiketimizin arkaplan olmasıdır.
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    #Burada torchvision models üzerinden maskrcnn_resnet50_fpn modelimizi indiriyoruz


    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):#Burada penfudandataset'e verdiğimiz transforms objemizin özelliklerini belirliyoruz
    transforms = []
    transforms.append(T.ToTensor())# PIL image 'i Pytorch Tensoruna çevirme özelliğini verdik
    if train:
        
        transforms.append(T.RandomHorizontalFlip(0.5))#resimleri rassal olarak döndürmesini sağlıyoruz
        #amacımız tabiki veri setinin çeşitliliğini arttırmak
    return T.Compose(transforms)

# dataseti tanımlıyoruz ve resimleri modifiye etmesini istiyoruz
dataset = PennFudanDataset(path, get_transform(train=True))
dataset_test = PennFudanDataset(path, get_transform(train=False))

datasetTestSize = int((datasetSize*30)/100)

print('Dataset Test Boyutu    :   ', datasetTestSize)
print('Dataset Eğitim Boyutu  :   ', datasetSize-datasetTestSize)

# dataseti ve test için kullanılacak dataseti ayarlıyoruz
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-datasetTestSize])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-datasetTestSize:])
#dataseti kardık ve teste 50 veri, geriye kalan veriyide datasetimize aktardık.

# training ve validation için dataloaderı ayarlıyoruz
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
#buradaki num_workers değeri eğitimi threadlere bölüyor.
#Eğer bir tane ekran kartımız var ise bunları 0 yapmalıyız
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#device = torch.device('cpu')
#Çalıştırılacak cihaz olarak uygunsa Gpu çalışması için cuda, uygun değilse cpu ayarlamasını sağlıyoruz.

num_classes = 2 #Daha öncede açıkladığım gibi 2 tane sınıfımız var biri arkaplan biri blob etiketimiz.

#pretrained modelimizi çekiyoruz
model = get_instance_segmentation_model(num_classes)
# modele uygun cihazda çalışması için bildiri yapıyoruz
model.to(device)


from PIL import Image
import torch
import torchvision.transforms as trans #torchvision'un transforms.py dosyasını çekiyoruz
import cv2
import numpy as np
device = torch.device('cuda')

#burada kaydettiğimiz modeli kullanmak için onu dışarıdan çağırıyoruz
#savedmodel = torch.load("/home/yonga/keremWorkSpace/BalıkSayma/FishCounterWithMaskRCNN/model.h5")
model.load_state_dict(torch.load("/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/TrainAndPrediction/maskRCNN_model_10.h5"))
savedmodel = model
#modeli evalation moduna alıyoruz burada device belirtmemiz gerekmedi bunu araştırmam gerekiyor
#kodumuz sorunsuz bir şekilde gpu 'da çalışıyor sanırım önceden modeli gpuda çalışacak şekilde-
#kaydettiğimiz için.
savedmodel.eval()

#Burada transforms'un ne yapacağını ona söylüyoruz
loader = trans.Compose([trans.ToTensor()])#loader resmi PIL image'den tensora çeviriyor

unloader = trans.ToPILImage() # unloader ise resmi tensordan PIL image'e çeviriyor


#Bu method image'i PIL image tipinde açar ve bunu tensor'a çevirir
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image)
    return image
#mainpath = '/home/shakabrah/keremGithubWorkspace/CanserCellsCounterWithMaskR-Cnn/TrainAndPrediction/deneme/Test/'
mainpath = '/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/Test/'
maskpath = mainpath+'masks/'
imagepath = mainpath+'images/'
### Version 2 tests paths
p1 = '7864.tiff'
p2 = '2609.tiff'
p3 = 'Basler_raL2048-48gm__22248034__20181106_144201677_0113.tiff'

imagepath = imagepath+p1
maskpath = maskpath+p1

image = cv2.imread(maskpath)

gercekObjeSayisi = np.unique(image).shape
print('Olması gereken sayı = ', gercekObjeSayisi)
tahminResmi = image_loader(imagepath)
torch.cuda.empty_cache()#gpu'nun cache'ini temizler, yer açar

import time
start = time.process_time()
with torch.no_grad():
    prediction = savedmodel([tahminResmi.to(device)])
print(time.process_time() - start)
objesayisi = prediction[0]['masks'].shape
objesayisi = objesayisi[0]

#print(prediction[0]['masks'])
print(prediction[0]['masks'].shape)
print("Obje Sayısı Tahminlerime Göre ",objesayisi, 'tane')

torch.save(model, "/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/TrainAndPrediction/model.h5")
