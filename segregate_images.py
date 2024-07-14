# class xcentre ycentre width height  <-- normalised values
############################## Computing accuracy using detect.py #######################################
import os
import shutil
import numpy as np
import argparse
from collections import Counter
from sklearn import preprocessing
from collections import OrderedDict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as precision_recall
import matplotlib.pyplot as plt

def create_dir(folder_path):
  if not os.path.isdir(folder_path):
    os.mkdir(folder_path)

# creates dictionary of unique class name with their conf score in descending order
def unique_labels(list_pred):
  d = {}
  for item in reversed(list_pred):
    class_num = item.split(' ')[0]
    conf = item.split(' ')[-1].split('\n')[0]
    # print(conf)
    if (int(class_num) not in d) and (float(conf) > 0.1): #conf threshold 
    # if (int(class_num) not in d): #conf threshold 
      d.update({int(class_num):conf})
    else:
      continue
  return d


parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, help='Testing images path')
parser.add_argument('--pred_path', type=str, help= 'Path to predicted labels from detect.py')
parser.add_argument('--segregated_path', type=str, help='Path to tagged images directory')
FLAGS = parser.parse_args()

# images_path = '/home/ashimag/share_iiit_raw_autoseg_testing_24-8-2022/'
# pred_path = '/home/ashimag/yolov5/runs/detect/yolo_test_24_08_correct_labels/labels/'
# # path of original images after segregation.
# segregated_folder_path = '/home/ashimag/segregated_images/'
# path of bbox images with predictions after segregation.
# segregated_folder_path_bbox = '/home/ashimag/segregated_images_bbox/'

# # 82 classes
# classes = ['mani_cras-Manis crassicaudata', 'maca_munz-Macaca munzala', 'maca_radi-Macaca radiata', 'athe_macr', 'vulp_beng', 'lept_java-Leptoptilos javanicus',
#  'trac_pile-Trachypithecus pileatus', 'hyst_brac-Hystrix brachyura', 'nilg_hylo-Nilgiritragus hylocrius', 'prio_vive-Prionailurus viverrinus',
#   'neof_nebu-Neofelis nebulosa', 'melu_ursi', 'vehi_vehi', 'hyae_hyae-Hyaena hyaena', 'maca_mula-Macaca mulatta', 'fran_pond-Francolinus pondicerianus',
#    'munt_munt-Muntiacus muntjak', 'feli_sylv-Felis sylvestris', 'maca_sile-Macaca silenus', 'vive_zibe-Viverra zibetha', 'rusa_unic-Rusa unicolor',
#     'lepu_nigr-Lepus nigricollis', 'vive_indi-Viverricula indica', 'pavo_cris', 'anti_cerv', 'gall_lunu-Galloperdix lunulata', 'cato_temm-Catopuma temminckii',
#      'sus__scro-Sus scrofa', 'cani_aure-Canis aureus', 'para_herm-Paradoxurus hermaphroditus', 'axis_axis', 'catt_kill', 'goat_sheep', 'vara_beng-Varanus bengalensis',
#       'para-jerd-Paradoxurus jerdoni', 'mart_gwat-Martes gwatkinsii', 'homo_sapi', 'semn_john+Semnopithecus johnii', 'herp_edwa-Herpestes edwardsii', 'bos__fron',
#        'herp_vitt-Herpestes vitticollis', 'arct_coll', 'dome_cats-Domestic cat', 'bos__indi', 'mell_cape-Mellivora capensis', 'ursu_thib-Ursus thibetanus',
#         'semn_ente-Semnopithecus entellus', 'prio_rubi-Prionailurus rubiginosus', 'dome_dogs-Domestic dog', 'cani_lupu-Canis lupus', 'gall_sonn-Gallus sonneratii',
#          'gaze_benn-Gazella bennettii', 'bose_trag-Boselaphus tragocamelus', 'budo_taxi-Budorcas taxicolor', 'bos__gaur', 'catt_catt-Cattle', 'blan_blan',
#           'cuon_alpi-Cuon alpinus', 'capr_thar-Capricornis thar', 'equu_caba-Equus caballus', 'herp_fusc-Herpestes fuscus', 'trac_john-Trachypithecus johnii',
#            'vara_salv-Varanus salvator', 'gall_gall-Gallus gallus', 'naem_gora-Naemorhedus goral', 'herp_urva-Herpestes urva', 'hyst_indi-Hystrix indica',
#             'herp_smit-Herpestes smithii', 'bird_bird', 'tetr_quad-Tetracerus quadricornis', 'feli_chau-Felis chaus', 'maca_arct-Macaca arctoides',
#              'lutr_pers-Lutrogale perspicillata', 'mosc_indi-Moschiola indica', 'pant_tigr', 'pant_pard-Panthera pardus', 'mart_flav-Martes flavigula',
#               'pagu_larv-Paguma larvata-Masked Palm Civet', 'prio_beng-Prionailurus bengalensis', 'gall_spad-Galloperdix spadicea', 'elep_maxi-Elephas maximus',
#                'axis_porc']

# # 82 classes
# destination_class_names = ['mani_cras', 'maca_munz', 'maca_radi', 'athe_macr', 'vulp_beng', 'lept_java','trac_pile', 'hyst_brac', 'nilg_hylo', 'prio_vive',
#   'neof_nebu', 'melu_ursi', 'vehi_vehi', 'hyae_hyae', 'maca_mula', 'fran_pond', 'munt_munt', 'feli_sylv', 'maca_sile', 'vive_zibe', 'rusa_unic', 'lepu_nigr',
#    'vive_indi', 'pavo_cris', 'anti_cerv', 'gall_lunu-', 'cato_temm', 'sus__scro', 'cani_aure', 'para_herm', 'axis_axis', 'catt_kill', 'goat_sheep',
#     'vara_beng', 'para-jerd', 'mart_gwat', 'homo_sapi', 'semn_john', 'herp_edwa', 'bos__fron', 'herp_vitt', 'arct_coll', 'dome_cats', 'bos__indi', 
#     'mell_cape', 'ursu_thib', 'semn_ente', 'prio_rubi', 'dome_dogs', 'cani_lupu', 'gall_sonn', 'gaze_benn', 'bose_trag', 'budo_taxi', 'bos__gaur', 
#         'catt_catt', 'blan_blan','cuon_alpi', 'capr_thar', 'equu_caba', 'herp_fusc', 'trac_john','vara_salv', 'gall_gall-Gallus gallus', 'naem_gora',
#          'herp_urva', 'hyst_indi', 'herp_smit', 'bird_bird', 'tetr_quad', 'feli_chau', 'maca_arct', 'lutr_pers', 'mosc_indi', 'pant_tigr', 'pant_pard',
#          'mart_flav', 'pagu_larv', 'prio_beng', 'gall_spad', 'elep_maxi', 'axis_porc']

# 98 classes
classes = ['mani_cras-Manis crassicaudata', 'maca_munz-Macaca munzala', 'maca_radi-Macaca radiata', 'athe_macr', 'vulp_beng', 'lept_java-Leptoptilos javanicus',
 'trac_pile-Trachypithecus pileatus', 'hyst_brac-Hystrix brachyura', 'nilg_hylo-Nilgiritragus hylocrius', 'prio_vive-Prionailurus viverrinus',
  'neof_nebu-Neofelis nebulosa', 'melu_ursi', 'vehi_vehi', 'hyae_hyae-Hyaena hyaena', 'maca_mula-Macaca mulatta', 'fran_pond-Francolinus pondicerianus',
   'munt_munt-Muntiacus muntjak', 'feli_sylv-Felis sylvestris', 'maca_sile-Macaca silenus', 'vive_zibe-Viverra zibetha', 'rusa_unic-Rusa unicolor',
    'lepu_nigr-Lepus nigricollis', 'vive_indi-Viverricula indica', 'pavo_cris', 'anti_cerv', 'gall_lunu-Galloperdix lunulata', 'cato_temm-Catopuma temminckii',
     'sus__scro-Sus scrofa', 'cani_aure-Canis aureus', 'para_herm-Paradoxurus hermaphroditus', 'axis_axis', 'catt_kill', 'goat_sheep', 'vara_beng-Varanus bengalensis',
      'para-jerd-Paradoxurus jerdoni', 'mart_gwat-Martes gwatkinsii', 'homo_sapi', 'semn_john+Semnopithecus johnii', 'herp_edwa-Herpestes edwardsii', 'bos__fron',
       'herp_vitt-Herpestes vitticollis', 'arct_coll', 'dome_cats-Domestic cat', 'bos__indi', 'mell_cape-Mellivora capensis', 'ursu_thib-Ursus thibetanus',
        'semn_ente-Semnopithecus entellus', 'prio_rubi-Prionailurus rubiginosus', 'dome_dogs-Domestic dog', 'cani_lupu-Canis lupus', 'gall_sonn-Gallus sonneratii',
         'gaze_benn-Gazella bennettii', 'bose_trag-Boselaphus tragocamelus', 'budo_taxi-Budorcas taxicolor', 'bos__gaur', 'catt_catt-Cattle', 'blan_blan',
          'cuon_alpi-Cuon alpinus', 'capr_thar-Capricornis thar', 'equu_caba-Equus caballus', 'herp_fusc-Herpestes fuscus', 'trac_john-Trachypithecus johnii',
           'vara_salv-Varanus salvator', 'gall_gall-Gallus gallus', 'naem_gora-Naemorhedus goral', 'herp_urva-Herpestes urva', 'hyst_indi-Hystrix indica',
            'herp_smit-Herpestes smithii', 'bird_bird', 'tetr_quad-Tetracerus quadricornis', 'feli_chau-Felis chaus', 'maca_arct-Macaca arctoides',
             'lutr_pers-Lutrogale perspicillata', 'mosc_indi-Moschiola indica', 'pant_tigr', 'pant_pard-Panthera pardus', 'mart_flav-Martes flavigula',
              'pagu_larv-Paguma larvata-Masked Palm Civet', 'prio_beng-Prionailurus bengalensis', 'gall_spad-Galloperdix spadicea', 'elep_maxi-Elephas maximus',
               'axis_porc', 'anat_elli', 'bats_bats', 'call_pyge-Callosciurus pygerythrus', 'came_came-Camel', 'capr_hisp-Caprolagus hispidus', 'funa_palm-Funambulus palmarum',
                'hela_mala-Helarctos malayanus', 'lutr_lutr-Lutra lutra', 'maca_assa-Macaca assamensis', 'maca_leon-Macaca leonina', 'maca_maca-Macaque', 
                'melo_pers', 'pard_marm-Pardofelis marmorata', 'prio_pard-Prionodon pardicolor', 'tree_shre', 'vulp_vulp']   # class names

destination_class_names = ['mani_cras', 'maca_munz', 'maca_radi', 'athe_macr', 'vulp_beng', 'lept_java','trac_pile', 'hyst_brac', 'nilg_hylo', 'prio_vive',
  'neof_nebu', 'melu_ursi', 'vehi_vehi', 'hyae_hyae', 'maca_mula', 'fran_pond', 'munt_munt', 'feli_sylv', 'maca_sile', 'vive_zibe', 'rusa_unic', 'lepu_nigr',
   'vive_indi', 'pavo_cris', 'anti_cerv', 'gall_lunu-', 'cato_temm', 'sus__scro', 'cani_aure', 'para_herm', 'axis_axis', 'catt_kill', 'goat_sheep',
    'vara_beng', 'para-jerd', 'mart_gwat', 'homo_sapi', 'semn_john', 'herp_edwa', 'bos__fron', 'herp_vitt', 'arct_coll', 'dome_cats', 'bos__indi', 
    'mell_cape', 'ursu_thib', 'semn_ente', 'prio_rubi', 'dome_dogs', 'cani_lupu', 'gall_sonn', 'gaze_benn', 'bose_trag', 'budo_taxi', 'bos__gaur', 
        'catt_catt', 'blan_blan','cuon_alpi', 'capr_thar', 'equu_caba', 'herp_fusc', 'trac_john','vara_salv', 'gall_gall-Gallus gallus', 'naem_gora',
         'herp_urva', 'hyst_indi', 'herp_smit', 'bird_bird', 'tetr_quad', 'feli_chau', 'maca_arct', 'lutr_pers', 'mosc_indi', 'pant_tigr', 'pant_pard',
         'mart_flav', 'pagu_larv', 'prio_beng', 'gall_spad', 'elep_maxi', 'axis_porc', 'anat_elli', 'bats_bats', 'call_pyge', 'came_came', 'capr_hisp',
          'funa_palm', 'hela_mala', 'lutr_lutr', 'maca_assa', 'maca_leon', 'maca_maca', 'melo_pers', 'pard_marm', 'prio_pard', 'tree_shre', 'vulp_vulp']   # class names

le = preprocessing.LabelEncoder()
le.fit(classes)
word_to_int = le.transform(classes)

res = dict(zip(classes, word_to_int))

# res['unid_unid'] = len(classes)

res_int_to_word = {}
keys = list(res.keys())
values = list(res.values())

for i in range(len(keys)):
    res_int_to_word[values[i]] = keys[i]

# sorted dictionary from int to word
sorted_res_int_to_word = OrderedDict(sorted(res_int_to_word.items()))

all_images = os.listdir(FLAGS.images_path)
all_pred_labels = os.listdir(FLAGS.pred_path)

# Check the original images provided and corresponding labels for those images.
print("Total images were {}, YOLO detected labels for {} images".format(len(all_images), len(all_pred_labels)))

############# segregate images in folders based on the detected label from txt files. 
# image_names = [file_name.split('.')[0] for file_name in all_images]
# first collect names of all images (remove the extension).
# create a dictionary mapping original image names with truncated names.
image_name_dict = {}
image_names = []
for file_name in all_images:
  image_name = file_name.split('.')[0]
  image_name_dict[image_name] = [file_name, '.'+file_name.split('.')[1]]      # save original file name and extension also
  image_names.append(image_name)

# iteratively look for predictions of these images and save preds in image_preds{} dictionary.
image_preds = {}
cnt = 0
for image in image_names:
  try:
    text_file = image + '.txt'
    file = open(os.path.join(FLAGS.pred_path, text_file), 'r')
    all_preds = file.readlines()
    pred_dict = unique_labels(all_preds)
    image_preds[image] = pred_dict
    # pred = all_preds[-1]
    # pred_class = pred.split(' ')[0]
    # image_preds[image] = int(pred_class)
  except :
    print('Generated label not found for this image {}'.format(image))
    # assign blan_blan label for the images whose predictions were not generated by yolo. 
    cnt = cnt + 1
    # may have to move these images in unid_unid folder later on.
    image_preds[image] = {}               # res['blan_blan']
    # image_preds[image] = res['unid_unid']

print("Total files without generated predictions are: ", cnt)
print("Created predictions dictionary")


# segregating images based on predictions using model.

# Create these destination directories if not already exists
create_dir(FLAGS.segregated_path)
# create_dir(segregated_folder_path_bbox)

# create destination directory dictionary
dest_dict = {}
for i, folder in enumerate(classes):
  dest_dict[folder] = destination_class_names[i]

def segregate_images(class_name, image):
  # class_name = sorted_res_int_to_word[image_preds[image]]
  folder_path = os.path.join(FLAGS.segregated_path, dest_dict[class_name])
  create_dir(folder_path)

  # folder_path_bbox = os.path.join(segregated_folder_path_bbox, class_name)
  # create_dir(folder_path_bbox)

  ### handle for all file extensions
  # src_image_path = image_name_dict[image][0]
  src_image_path = os.path.join(FLAGS.images_path, image + image_name_dict[image][1]) 
  dest_image_path = os.path.join(folder_path, image + image_name_dict[image][1])
  shutil.copy(src_image_path, dest_image_path)

cnt = 0
for image in image_preds:
  pred_dict = image_preds[image]
  # for empty label.txt file, put the image in blank folder
  # move them to blank blank folders. 
  if len(pred_dict) == 0:
    class_name = 'blan_blan'
    # print("blank image: ", image)
    # import pdb; pdb.set_trace()
    segregate_images(class_name, image)

  elif len(pred_dict.keys()) <= 3: 
    index = 0
    for label, conf in pred_dict.items():
      class_name = sorted_res_int_to_word[label]
      if index == 2 or index == 1:
        if class_name == 'vehi_vehi' or class_name == 'homo_sapi':
          # print(image)
          # print(pred_dict)
          # import pdb; pdb.set_trace()
          segregate_images(class_name, image)
      else:
        segregate_images(class_name, image)
      index = index + 1

  elif len(pred_dict.keys()) > 3:
    count = 1
    index = 0
    for label, conf in pred_dict.items():
      class_name = sorted_res_int_to_word[label]
      if index == 2 or index == 1:
        if class_name == 'vehi_vehi' or class_name == 'homo_sapi':
          segregate_images(class_name, image)
      else:
        segregate_images(class_name, image)
      count += 1
      if count > 3:
        break
      index = index + 1

  cnt = cnt + 1
  print(cnt)


# keyword.append(f'{prefix[index]}_{mapping[sorted_res_int_to_word[label]]}')
# key_with_conf.append(f'{prefix[index]}_{mapping[sorted_res_int_to_word[label]]}_{conf}')
        
# keyword.append(f'{prefix[index]}_{mapping[sorted_res_int_to_word[label]]}')
# key_with_conf.append(f'{prefix[index]}_{mapping[sorted_res_int_to_word[label]]}_{conf}')
# # src_image_path = image_name_dict[image]
# src_image_path = os.path.join(images_path_bbox, image + image_name_dict[image][1]) 
# # src_image_path = os.path.join(images_path_bbox, image + '.jpg')
# dest_image_path = os.path.join(folder_path_bbox, image + image_name_dict[image][1])
# shutil.copy(src_image_path, dest_image_path)
  