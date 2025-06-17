from utils.data_pipeline import VisualReviewer
from utils.data_pipeline import DetectionFilter
from utils.dataset_pipeline import LabelStudioConverter
from utils.dataset_pipeline import LabelUpdater
from utils.dataset_pipeline import DatasetMerger
from utils.data_pipeline import DataAugmentor
from utils.dataset_pipeline import DatasetWithYamlAsZip
from utils.config import get_model_path, get_dataset_path, get_label_studio_export_path, get_label_studio_import_path, get_video_path, get_detection_params


def prepare_new_dataset_and_labelStudio():
    # Get detection parameters for player
    detection_params = get_detection_params("player")
    
    detector = DetectionFilter(
        yolo_model_path=get_model_path("player"),
        confidence_threshold=detection_params['confidence_threshold'],
        frame_interval=detection_params['frame_interval']
    )
    detector.create_detection_dataset(
        video_path=get_video_path(),
        dataset_output_path=get_dataset_path("player"),
        target_class_id=0,
        min_detections=detection_params['min_detections'],
        max_detections=detection_params['max_detections']
    )

    reviewer = VisualReviewer(class_names=['player'])
    for split in ['train', 'valid']:
        reviewer.review_annotated_frames(
            dataset_dir=get_dataset_path("player"),
            split=split
        )

    # Convert for Label Studio
    for split in ['train', 'valid']:
        converter = LabelStudioConverter(
            type="player",
            split=split,
            data_dir=get_dataset_path("player", f"rejected/{split}"),
            output_json=get_label_studio_export_path("player", split)
        )
        converter.convert()


def update_new_dataset():
    updater = LabelUpdater(
        new_label_dir=get_label_studio_import_path("player"),
        dataset_dir=get_dataset_path("player", "rejected")
    )
    updater.update()

    merger = DatasetMerger(
        parent_dataset_dir=get_dataset_path("player"),
        other_dataset_dir=get_dataset_path("player", "rejected"),
        decider="remove"
    )
    merger.merge()

    augmentor = DataAugmentor(augmentations_per_image=3)
    for split in ['train', 'valid']:
        augmentor.augment_dataset(
            input_img_dir=get_dataset_path("player", f"{split}/images"),
            input_lbl_dir=get_dataset_path("player", f"{split}/labels"),
            output_img_dir=get_dataset_path("player", f"{split}/images"),
            output_lbl_dir=get_dataset_path("player", f"{split}/labels")
        )
    
    zipper = DatasetWithYamlAsZip(
        folder_path=get_dataset_path("player"),
        class_name="player"
    )
    zipper.zip_folder_with_yaml()


if __name__ == "__main__":
    update_new_dataset() 