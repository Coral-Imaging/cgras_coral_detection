import os
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

class SegmentationValidator:
    def __init__(self, model_path, yaml_path, output_dir, conf_thresh=0.5, iou_thresh=0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset paths from YAML
        with open(yaml_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.val_images = [os.path.join(self.data_config['path'], path) for path in self.data_config['val']]
        self.class_names = self.data_config['names']
        self.class_names[4] = "background"  # Adding background class
        self.num_classes = len(self.class_names)

        # Create output subfolders
        for category in ['TP', 'FP', 'FN', 'TN']:
            os.makedirs(os.path.join(output_dir, category), exist_ok=True)

    def compute_iou(self, mask_pred, mask_gt):
        intersection = np.logical_and(mask_pred, mask_gt).sum()
        union = np.logical_or(mask_pred, mask_gt).sum()
        return intersection / union if union > 0 else 0.0
    
    def validate(self):
        results = []
        confusion_data = np.zeros((self.num_classes, self.num_classes), dtype=int)

        for img_dir in self.val_images:
            label_dir = img_dir.replace('/images', '/labels')
            images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
            
            for img_file in images:
                img_path = os.path.join(img_dir, img_file)
                label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                
                if not os.path.exists(label_path):
                    continue
                
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                preds = self.model(img_path, conf=self.conf_thresh)[0]
                
                pred_masks = {cls: np.zeros(img.shape[:2], dtype=np.uint8) for cls in range(self.num_classes)}
                if preds.masks is not None and preds.boxes is not None:
                    for mask, box in zip(preds.masks.xy, preds.boxes.cls):
                        cls = int(box.item())
                        cv2.fillPoly(pred_masks[cls], [mask.astype(np.int32)], 1)
                
                gt_masks = {cls: np.zeros(img.shape[:2], dtype=np.uint8) for cls in range(self.num_classes)}
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        class_id, *coords = map(float, line.strip().split())
                        polygon = np.array(coords).reshape(-1, 2) * np.array([img.shape[1], img.shape[0]])
                        polygon = polygon.astype(np.int32)
                        cv2.fillPoly(gt_masks[int(class_id)], [polygon], 1)
                
                for cls in range(self.num_classes):
                    iou = self.compute_iou(pred_masks[cls], gt_masks[cls])
                    pred_class = cls if pred_masks[cls].sum() > 0 else 4  # Assign background if no prediction
                    gt_class = cls if gt_masks[cls].sum() > 0 else 4  # Assign background if no ground truth
                    confusion_data[gt_class, pred_class] += 1
                    
                    category = 'TP' if (pred_class == cls and gt_class == cls) else \
                               'FP' if (pred_class == cls and gt_class != cls) else \
                               'FN' if (pred_class != cls and gt_class == cls) else 'TN'
                    
                    save_path = os.path.join(self.output_dir, category, f"{img_file}_cls{cls}.png")
                    self.visualize_and_save(img_rgb, gt_masks[cls], pred_masks[cls], save_path, cls)
                
                results.append([img_file])
        
        self.plot_confusion_matrix(confusion_data)
        print("Validation complete!")
        
    def visualize_and_save(self, img, gt_mask, pred_mask, save_path, cls):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].imshow(gt_mask, alpha=0.5, cmap='jet')
        ax[0].set_title(f"Ground Truth: {self.class_names[cls]}")
        ax[0].axis("off")
        
        ax[1].imshow(img)
        ax[1].imshow(pred_mask, alpha=0.5, cmap='jet')
        ax[1].set_title(f"Prediction: {self.class_names[cls]}")
        ax[1].axis("off")
        
        plt.savefig(save_path)
        plt.close()
    
    def plot_confusion_matrix(self, matrix):
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names.values(), yticklabels=self.class_names.values())
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.title("Segmentation Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

# Example usage
if __name__ == "__main__":

    SSD_PATH = "/media/java/RRAP03"
    REPO_PATH = "/home/java/repos/cgras_coral_detection"
    TRAIN = "train7"
    CONF_THRESH = 0.25
    IOU_THRESH = 0.6

    model_path = os.path.join(REPO_PATH, "segmenter/cgras_segmentation", TRAIN, "weights/best.pt")
    yaml_path = os.path.join(SSD_PATH, "outputs/train/cgras_data.yaml")
    output_dir = os.path.join(SSD_PATH, "outputs/validation_results")

    validator = SegmentationValidator(
        model_path=model_path,
        yaml_path=yaml_path,
        output_dir=output_dir,
        conf_thresh=CONF_THRESH,
        iou_thresh=IOU_THRESH
    )
    validator.validate()
    print("Validation complete!")
