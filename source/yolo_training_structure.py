import shutil

def distribute_train_val_files(file_list, training_ids, valid_ids, data_dir, train_dir, val_dir, full_path=False):

    for path in file_list:
        if full_path == True:
            file = path.split('/')[-1]
            #print(file)
        else: 
            file = path
        #print(file)
        img_id = int(file.split('_')[-1].split('.')[0])
        #print(img_id)
        #print(type(img_id))
        if img_id in training_ids:
            #print('train')
            source_path_img = data_dir / file
            destination_path_img = train_dir / file
            shutil.copy(source_path_img, destination_path_img)
            
        elif img_id in valid_ids:
            #print('val')
            source_path = data_dir / file
            destination_path = val_dir / file
            shutil.copy(source_path, destination_path)