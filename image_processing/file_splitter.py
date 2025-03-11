import os
import shutil
import glob
import random

class DatasetSplitter:
    def __init__(self, data_location, save_dir, train_ratio=0.70, test_ratio=0.15, valid_ratio=0.15, max_files=16382):
        self.data_location = data_location
        self.save_dir = save_dir
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        self.max_files = max_files
        
        self._validate_ratios()
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.image_list = sorted(glob.glob(os.path.join(self.data_location, 'images', '*.jpg')))
        self.label_list = sorted(glob.glob(os.path.join(self.data_location, 'labels', '*.txt')))
        self.total_files = len(self.label_list)

    def _validate_ratios(self):
        if not (0 <= self.test_ratio <= 1 and 0 <= self.train_ratio <= 1 and 0 <= self.valid_ratio <= 1):
            raise ValueError("All ratios must be between 0 and 1.")
        if not (self.train_ratio + self.test_ratio + self.valid_ratio == 1):
            raise ValueError("The sum of train, validation, and test ratios must equal 1.")
    
    def _separate_files(self, number, img_list, txt_list, old_img_list, old_txt_list):
        for _ in range(int(number)):
            r = random.randint(0, len(old_txt_list) - 1)
            img_list.append(old_img_list.pop(r))
            txt_list.append(old_txt_list.pop(r))
        return old_img_list, old_txt_list
    
    def _copy_link(self, src, dst):
        if os.path.islink(src):
            os.symlink(os.readlink(src), os.path.join(dst, os.path.basename(src)))
        else:
            shutil.copy(src, dst)
    
    def _move_files(self, file_list, save_path, sub_folder):
        output_path = os.path.join(save_path, sub_folder)
        os.makedirs(output_path, exist_ok=True)
        for item in file_list:
            self._copy_link(item, output_path)
    
    def _split_and_move_files(self, file_list, image_list, prefix):
        if len(file_list) >= self.max_files:
            num_splits = len(file_list) // self.max_files
            for i in range(num_splits):
                split_files, split_images = file_list[:self.max_files], image_list[:self.max_files]
                file_list, image_list = file_list[self.max_files:], image_list[self.max_files:]
                self._move_files(split_files, self.save_dir, f'{prefix}_{i}/labels')
                self._move_files(split_images, self.save_dir, f'{prefix}_{i}/images')
            self._move_files(file_list, self.save_dir, f'{prefix}_{num_splits}/labels')
            self._move_files(image_list, self.save_dir, f'{prefix}_{num_splits}/images')
        else:
            self._move_files(file_list, self.save_dir, f'{prefix}/labels')
            self._move_files(image_list, self.save_dir, f'{prefix}/images')
    
    def split_dataset(self):
        valid_images, valid_labels, test_images, test_labels = [], [], [], []
        remaining_images, remaining_labels = self.image_list, self.label_list
        
        remaining_images, remaining_labels = self._separate_files(self.total_files * self.valid_ratio, valid_images, valid_labels, remaining_images, remaining_labels)
        remaining_images, remaining_labels = self._separate_files(self.total_files * self.test_ratio, test_images, test_labels, remaining_images, remaining_labels)
        
        self._split_and_move_files(remaining_labels, remaining_images, 'train')
        self._split_and_move_files(valid_labels, valid_images, 'valid')
        self._split_and_move_files(test_labels, test_images, 'test')

# Example usage
if __name__ == "__main__":
    data_location = "/media/agoni/RRAP03/tiled_dataset"
    save_dir = "/media/agoni/RRAP03/split_dataset"
    
    splitter = DatasetSplitter(data_location, save_dir)
    splitter.split_dataset()