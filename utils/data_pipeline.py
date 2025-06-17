import os
import cv2
import shutil
import random
import albumentations as A
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataExtractor:
    def __init__(self, dataset_root, output_root, target_class_id, remap_to_class_0=True):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.target_class_id = target_class_id
        self.remap_to_class_0 = remap_to_class_0

    def extract(self):
        """Extract and filter data for the target class"""
        for subset in ["train", "valid"]:
            label_dir = self.dataset_root / subset / "labels"
            image_dir = self.dataset_root / subset / "images"

            out_label_dir = self.output_root / subset / "labels"
            out_image_dir = self.output_root / subset / "images"

            os.makedirs(out_label_dir, exist_ok=True)
            os.makedirs(out_image_dir, exist_ok=True)

            for filename in os.listdir(label_dir):
                if not filename.endswith(".txt"):
                    continue

                label_path = label_dir / filename
                with open(label_path, "r") as f:
                    lines = f.readlines()

                filtered_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts[0] == str(self.target_class_id):
                        if self.remap_to_class_0:
                            parts[0] = "0"
                        filtered_lines.append(" ".join(parts) + "\n")

                if filtered_lines:
                    out_label_path = out_label_dir / filename
                    with open(out_label_path, "w") as f:
                        f.writelines(filtered_lines)

                    base_name = os.path.splitext(filename)[0]
                    for ext in [".jpg", ".jpeg", ".png"]:
                        image_path = image_dir / (base_name + ext)
                        if image_path.exists():
                            shutil.copy(image_path, out_image_dir / (base_name + ext))
                            break

                    print(f"✔ {subset}/{filename} → kept {len(filtered_lines)} annotations")
                else:
                    print(f"⚠ {subset}/{filename} → no class {self.target_class_id}, skipped")

class DetectionFilter:
    def __init__(self, yolo_model_path, confidence_threshold=0.25, frame_interval=60):
        self.model = YOLO(yolo_model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_interval = frame_interval

    def create_detection_dataset(
        self,
        video_path,
        dataset_output_path,
        target_class_id= 0,
        min_detections= None,
        max_detections= None,
        split_ratios=(0.85, 0.15)
    ):
        video_name = Path(video_path).stem
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        saved_frames = []

        temp_img_dir = Path(dataset_output_path) / "temp_images"
        temp_lbl_dir = Path(dataset_output_path) / "temp_labels"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        temp_lbl_dir.mkdir(parents=True, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_interval == 0:
                results = self.model(frame)[0]
                detections = [
                    det for det in results.boxes.data.cpu().numpy()
                    if det[4] >= self.confidence_threshold and int(det[5]) == target_class_id
                ]

                if len(detections) >= min_detections:
                    top_detections = sorted(detections, key=lambda d: d[4], reverse=True)[:max_detections]

                    h, w = frame.shape[:2]
                    img_filename = f"{video_name}_frame_{frame_idx:05d}.jpg"
                    lbl_filename = f"{video_name}_frame_{frame_idx:05d}.txt"

                    img_path = temp_img_dir / img_filename
                    lbl_path = temp_lbl_dir / lbl_filename

                    cv2.imwrite(str(img_path), frame)
                    with open(lbl_path, "w") as f:
                        for det in top_detections:
                            x1, y1, x2, y2, conf, cls = det
                            x_center = ((x1 + x2) / 2) / w
                            y_center = ((y1 + y2) / 2) / h
                            bbox_w = (x2 - x1) / w
                            bbox_h = (y2 - y1) / h
                            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}\n")

                    saved_frames.append(img_path)

            frame_idx += 1

        cap.release()
        print(f"✅ {len(saved_frames)} frames saved for video '{video_name}'.")

        # Dataset split
        valid_split = split_ratios[1] / (split_ratios[0] + split_ratios[1]) if (split_ratios[0] + split_ratios[1]) > 0 else 0.0

        #train_val, test = train_test_split(saved_frames, test_size=test_split, random_state=42) if test_split > 0 else (saved_frames, [])
        train, valid = train_test_split(saved_frames, test_size=valid_split, random_state=42)

        splits = {'train': train, 'valid': valid}

        for split_name, split_imgs in splits.items():
            img_out_dir = Path(dataset_output_path) / split_name / "images"
            lbl_out_dir = Path(dataset_output_path) / split_name / "labels"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            lbl_out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_imgs:
                lbl_path = temp_lbl_dir / (img_path.stem + ".txt")
                shutil.move(str(img_path), img_out_dir / img_path.name)
                if lbl_path.exists():
                    shutil.move(str(lbl_path), lbl_out_dir / lbl_path.name)

        # Try to remove the original directory
        try:
            shutil.rmtree(temp_img_dir)
            shutil.rmtree(temp_lbl_dir)
            print(f"✅ Removed original directory: {temp_img_dir} and {temp_lbl_dir}")
        except Exception as e:
            print(f"⚠️ Could not remove {temp_lbl_dir} or {temp_img_dir}: {str(e)}")

        print("📊 Split complete for this video:")
        print(f" - Train: {len(train)} images")
        print(f" - Valid: {len(valid)} images")


class VisualReviewer:
    def __init__(self, class_names=None, max_display_size=(1280, 720)):
        self.class_names = class_names
        self.max_display_size = max_display_size

    def review_annotated_frames(
        self,
        dataset_dir,
        split=None,
        filename_prefix=None
    ):
        dataset_dir = Path(dataset_dir)
        if split:
            image_dir = dataset_dir / split / "images"
            label_dir = dataset_dir / split / "labels"
            rejected_image_dir = dataset_dir / "rejected" / split / "images"
            rejected_label_dir = dataset_dir / "rejected" / split / "labels"
        else:
            image_dir = dataset_dir / "images"
            label_dir = dataset_dir / "labels"
            rejected_image_dir = dataset_dir / "rejected" / "images"
            rejected_label_dir = dataset_dir / "rejected" / "labels"

        rejected_image_dir.mkdir(parents=True, exist_ok=True)
        rejected_label_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(image_dir.glob("*.jpg"))
        if filename_prefix:
            image_files = [f for f in image_files if f.name.startswith(filename_prefix)]

        for image_path in image_files:
            label_path = label_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                continue

            original_h, original_w = image.shape[:2]

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id, x, y, bw, bh = map(float, parts)
                    x1 = int((x - bw / 2) * original_w)
                    y1 = int((y - bh / 2) * original_h)
                    x2 = int((x + bw / 2) * original_w)
                    y2 = int((y + bh / 2) * original_h)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = self.class_names[int(cls_id)] if self.class_names else str(int(cls_id))
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            max_w, max_h = self.max_display_size
            scale = min(max_w / original_w, max_h / original_h, 1.0)
            if scale < 1.0:
                image = cv2.resize(image, (int(original_w * scale), int(original_h * scale)))

            cv2.imshow("Review Frame (y: keep, n: move to rejected, ESC: quit)", image)
            key = cv2.waitKey(0)

            if key == ord('y'):
                print(f"✅ Keeping {image_path.name}")
                continue
            elif key == ord('n'):
                print(f"🚫 Moving {image_path.name} and label to rejected/")
                shutil.move(str(image_path), rejected_image_dir / image_path.name)
                if label_path.exists():
                    shutil.move(str(label_path), rejected_label_dir / label_path.name)
            elif key == 27:  # ESC
                print("❌ Stopped reviewing early.")
                break

        cv2.destroyAllWindows()


class DataAugmentor:
    def __init__(self, augmentations_per_image=3):
        self.augmentations_per_image = augmentations_per_image
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=5, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.2),
            A.MotionBlur(p=0.1),
            A.Resize(640, 640),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def load_yolo_labels(self, label_path):
        boxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(float(parts[0]))
                    bbox = list(map(float, parts[1:]))
                    boxes.append(bbox)
                    class_labels.append(class_id)
        return boxes, class_labels

    def augment_dataset(self, input_img_dir, input_lbl_dir, output_img_dir, output_lbl_dir):
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_lbl_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.png'))]

        for img_name in tqdm(image_files, desc="Augmenting"):
            img_path = os.path.join(input_img_dir, img_name)
            lbl_path = os.path.join(input_lbl_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

            if not os.path.exists(lbl_path):
                continue

            image = cv2.imread(img_path)
            boxes, class_labels = self.load_yolo_labels(lbl_path)

            for i in range(self.augmentations_per_image):
                try:
                    augmented = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
                    aug_img = augmented['image']
                    aug_boxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']

                    if not aug_boxes:
                        continue

                    aug_img_name = img_name.replace('.jpg', f'_aug{i}.jpg').replace('.png', f'_aug{i}.png')
                    aug_lbl_name = aug_img_name.replace('.jpg', '.txt').replace('.png', '.txt')

                    cv2.imwrite(os.path.join(output_img_dir, aug_img_name), aug_img)

                    with open(os.path.join(output_lbl_dir, aug_lbl_name), 'w') as f:
                        for cls, box in zip(aug_labels, aug_boxes):
                            f.write(f"{cls} {' '.join([f'{x:.6f}' for x in box])}\n")
                except Exception as e:
                    print(f"Error augmenting {img_name}: {e}")


# Example usage:
if __name__ == "__main__":
    # Example for player detection
    extractor = DataExtractor(
        dataset_root="../Waterpolo.v2i.yolov8",
        output_root="filtered_dataset",
        target_class_id=3,
        remap_to_class_0=True
    )
    extractor.extract()

    # Example for ball detection
    detector = DetectionFilter(
        yolo_model_path="../models/best_player_v1.pt",
        confidence_threshold=0.25,
        frame_interval=60
    )
    detector.create_detection_dataset(
        video_path="../videos/added_to_dataset/Q3.mp4",
        dataset_output_path="datasets/waterpolo_ball_dataset",
        target_class_id=0,
        min_detections=1,
        max_detections=1
    )

    # Example for visual review
    reviewer = VisualReviewer(class_names=['ball', 'player', 'goalkeeper'])
    reviewer.review_annotated_frames(
        dataset_dir="datasets/waterpolo_ball_dataset",
        split="test",
        filename_prefix="Q3"
    )

    # Example for augmentation
    augmentor = DataAugmentor(augmentations_per_image=3)
    augmentor.augment_dataset(
        input_img_dir='player_dataset_v1/valid/images',
        input_lbl_dir='player_dataset_v1/valid/labels',
        output_img_dir='player_dataset_v2/valid/images',
        output_lbl_dir='player_dataset_v2/valid/labels'
    ) 