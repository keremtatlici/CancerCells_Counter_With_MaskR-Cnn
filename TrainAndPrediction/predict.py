from PIL import Image
import torch
import torchvision.transforms as trans #torchvision'un transforms.py dosyasını çekiyoruz

device = torch.device('cuda')

#burada kaydettiğimiz modeli kullanmak için onu dışarıdan çağırıyoruz
#savedmodel = torch.load("/home/yonga/keremWorkSpace/BalıkSayma/FishCounterWithMaskRCNN/model.h5")
savedmodel = torch.load("model.h5")
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


### Version 2 tests paths
p1 = '/home/yonga/keremWorkSpace/CancerCellsCounterWithMaskR-Cnn/Dataset/Test/images/7862.tiff'
p2 = '/home/shakabrah/keremWorkSpace/BalıkSaymav2/HamsilerWithEvolation/Dataset4MaskRCNN/Imgs/Basler_raL2048-48gm__22248034__20181106_144201677_0287.tiff'
p3 = '/home/shakabrah/keremWorkSpace/BalıkSaymav2/HamsilerWithEvolation/Dataset4MaskRCNN/Imgs/Basler_raL2048-48gm__22248034__20171011_121137498_0008.tiff'

tahminResmi = image_loader(p1)
torch.cuda.empty_cache()#gpu'nun cache'ini temizler, yer açar

import time
start = time.process_time()
with torch.no_grad():
    prediction = savedmodel([tahminResmi.to(device)])
print(time.process_time() - start)
baliksayisi = prediction[0]['masks'].shape
baliksayisi = baliksayisi[0]
print("Obje Sayısı Tahminlerime Göre ",baliksayisi, 'tane')