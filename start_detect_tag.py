import os
import sys

# print(sys.argv)
# model_folder= sys.argv[1]
model_folder = os.path.dirname(os.path.abspath(__file__))
# import pdb; pdb.set_trace()
image_folder = sys.argv[1]
# final_folder = sys.argv[2]
final_folder= os.path.join(os.path.dirname(image_folder), os.path.basename(image_folder)+'_results')
if not os.path.exists(final_folder):
    os.mkdir(final_folder)
segregation_folder = sys.argv[2]
# remove_tags= sys.argv[3]

detect = os.path.join(model_folder ,"yolov5", "detect.py")
best_model = os.path.join(model_folder,"yolov5","models", "best_28_072.pt")
yaml = os.path.join(model_folder,"yolov5","data","wii_aite_2022_testing.yaml")
tag_py = os.path.join(model_folder,"tag_images.py")
segregate_py = os.path.join(model_folder, "segregate_images.py")
rm_tags_py=os.path.join(model_folder, "remove_tags.py")
f = os.walk(image_folder).__next__()[1]
# print(f)
name= os.path.basename(image_folder)
test_dir = os.path.join(final_folder, name)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# c=0
# print(f'Counter is {c}')
if len(f)==0:
    print('No subfolder inside the images path provided. It should have atleast one subfolder.')

log_folder=os.path.join(os.path.normpath(image_folder + os.sep + os.pardir), 'logs_files')
print(log_folder)
if not os.path.exists(log_folder):
    os.mkdir(log_folder)
if not os.path.exists(os.path.join(log_folder,'corrupt_images')):
    os.mkdir(os.path.join(log_folder,'corrupt_images'))

# Create empty detected_files.txt
if not os.path.isfile(os.path.join(log_folder, 'detected_files.txt')):
    with open(os.path.join(log_folder, 'detected_files.txt'), 'w') as fp:
        pass

for i in f:
    print("Processing {} subfolder: ".format(i))
    src_path = os.path.join(image_folder, i)
    # dest_path = os.path.join(segregation_folder, i)
    # print(src_path)
    # current_test = 'tested'+str(c)
    current_test=os.path.basename(src_path)
    # dest_path = os.mkdir(os.path.join(test_dir, current_test))
    #run detection code with source path given as src_path and destination as dest_path
    # print('Here')
    # s = 'python home/ashimag/WII_animal_detection/yolov5/detect.py --source %s --weights home/ashimag/Model_checkpoint/best.pt --data home/ashimag/WII_animal_detection/yolov5/data/wii_aite_2022_testing.yaml --img 640 --save-txt --save-conf --project %s --name %s  --conf-thres 0.001 --iou-thres 0.6' % (src_path,test_dir,current_test)
    # 
    s = 'python %s --source %s --weights %s --data %s --img 640 --save-txt --save-conf --project %s --name %s  --conf-thres 0.01 --nosave --iou-thres 0.6 --resume --empty_path %s' % (detect,src_path, best_model,yaml, test_dir,current_test, os.path.join(log_folder,'corrupt_images'))
    rm_tags= 'python %s --images_path %s '%(rm_tags_py,src_path)
    # segregate = 'python %s --images_path %s --pred_path %s --segregated_path %s' %(segregate_py, src_path, os.path.join(test_dir, current_test)+'\\labels', dest_path)
    # tag = 'python %s --images_path %s --pred_path %s --tagged_path %s' %(tag_py, src_path, os.path.join(test_dir, current_test)+'\\labels', src_path)
    tag = 'python %s --images_path %s --pred_path %s --tagged_path %s' %(tag_py, src_path, os.path.join(test_dir, current_test, 'labels'), segregation_folder)
    
    os.system(s)
    # os.system(segregate)
    # print("Segregation Done!")
    # print(remove_tags)
    # if remove_tags=='Yes':
    #     os.system(rm_tags)
    #     print('Removal of tags done')
    print('################################################')
    os.system(tag)
    print('################################################')
    print('Detection and tagging Done')
    print('################################################')
   
    # c+=1
