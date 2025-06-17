from utils.data_pipeline import VisualReviewer
from utils.data_pipeline import DetectionFilter
from utils.dataset_pipeline import LabelStudioConverter
from utils.dataset_pipeline import LabelUpdater
from utils.dataset_pipeline import DatasetMerger
from utils.data_pipeline import DataAugmentor
from utils.config import get_model_path, get_dataset_path, get_label_studio_export_path, get_label_studio_import_path, get_video_path, get_detection_params


def prepare_new_dataset_and_labelStudio():
    # Get detection parameters for goal
    detection_params = get_detection_params("goal")
    
    detector = DetectionFilter(
        yolo_model_path=get_model_path("goal"),
        confidence_threshold=detection_params['confidence_threshold'],
        frame_interval=detection_params['frame_interval']
    )
    detector.create_detection_dataset(
        video_path=get_video_path(),
        dataset_output_path=get_dataset_path("goal"),
        target_class_id=0,
        min_detections=detection_params['min_detections'],
        max_detections=detection_params['max_detections']
    )

    reviewer = VisualReviewer(class_names=['goal'])
    for split in ['test', 'train', 'valid']:
        reviewer.review_annotated_frames(
            dataset_dir=get_dataset_path("goal"),
            split=split
        )

    # Convert for Label Studio
    for split in ['train', 'valid']:
        converter = LabelStudioConverter(
            type="goal",
            data_dir=get_dataset_path("goal", f"rejected/{split}"),
            output_json=get_label_studio_export_path("goal", split)
        )
        converter.convert()


def update_new_dataset():
    updater = LabelUpdater(
        new_label_dir=get_label_studio_import_path("goal"),
        dataset_dir=get_dataset_path("goal", "rejected")
    )
    updater.update()

    merger = DatasetMerger(
        parent_dataset_dir=get_dataset_path("goal"),
        other_dataset_dir=get_dataset_path("goal", "rejected"),
        decider="remove"
    )
    merger.merge()

    augmentor = DataAugmentor(augmentations_per_image=3)
    for split in ['train', 'valid']:
        augmentor.augment_dataset(
            input_img_dir=get_dataset_path("goal", f"{split}/images"),
            input_lbl_dir=get_dataset_path("goal", f"{split}/labels"),
            output_img_dir=get_dataset_path("goal", f"{split}/images"),
            output_lbl_dir=get_dataset_path("goal", f"{split}/labels")
        )


if __name__ == "__main__":
    update_new_dataset() 