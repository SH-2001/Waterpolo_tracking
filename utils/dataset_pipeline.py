import os
import json
import shutil
import subprocess
import requests
from pathlib import Path
import zipfile
import yaml


class DatasetMerger:
    def __init__(self, parent_dataset_dir, other_dataset_dir,decider):
        self.parent_dataset_dir = Path(parent_dataset_dir)
        self.other_dataset_dir = Path(other_dataset_dir)
        self.decider = decider
        self.splits = ['train', 'valid', 'test']

    def merge(self):
        """Merge two datasets, moving files from other_dataset to parent_dataset"""
        for split in self.splits:
            # Paths for images and labels in parent dataset
            parent_img_dir = self.parent_dataset_dir / split / 'images'
            parent_label_dir = self.parent_dataset_dir / split / 'labels'

            # Paths for images and labels in other dataset
            other_img_dir = self.other_dataset_dir / split / 'images'
            other_label_dir = self.other_dataset_dir / split / 'labels'

            # Make sure parent directories exist
            parent_img_dir.mkdir(parents=True, exist_ok=True)
            parent_label_dir.mkdir(parents=True, exist_ok=True)

            # Move images
            if other_img_dir.exists():
                for img_file in other_img_dir.glob('*'):
                    dst_img_path = parent_img_dir / img_file.name
                    if not dst_img_path.exists():
                        if self.decider == "remove":
                            shutil.move(img_file, dst_img_path)
                        else:
                            shutil.copy(img_file,dst_img_path)
                    else:
                        print(f"Image file {img_file.name} already exists in {split} images, skipping or rename manually.")

            # Move labels
            if other_label_dir.exists():
                for label_file in other_label_dir.glob('*.txt'):
                    dst_label_path = parent_label_dir / label_file.name
                    if not dst_label_path.exists():
                        if self.decider == "remove":
                            shutil.move(label_file, dst_label_path)
                        else:
                            shutil.copy(label_file,dst_label_path)
                    else:
                        print(f"Label file {label_file.name} already exists in {split} labels, skipping or rename manually.")

        # Try to remove the other dataset directory
        if self.decider == "remove":
            try:
                shutil.rmtree(self.other_dataset_dir)
                print(f"✅ Removed other dataset directory: {self.other_dataset_dir}")
            except Exception as e:
                print(f"⚠️ Could not remove {self.other_dataset_dir}: {str(e)}")


class LabelUpdater:
    def __init__(self, new_label_dir, dataset_dir):
        self.new_label_dir = Path(new_label_dir)
        self.dataset_dir = Path(dataset_dir)
        self.subsets = ['train', 'valid', 'test']

    def update(self):
        """Update existing YOLO labels with new labels from Label Studio"""
        num_updated = 0
        not_found = []

        for label_file in self.new_label_dir.glob('*.txt'):
            # Remove prefix up to and including first dash
            if "-" in label_file.name:
                _, stripped_name = label_file.name.split("-", 1)
            else:
                stripped_name = label_file.name

            label_basename = Path(stripped_name).stem

            # Find which split contains the matching image
            found = False
            for split in self.subsets:
                image_path = self.dataset_dir / split / "images" / f"{label_basename}.jpg"
                label_dest = self.dataset_dir / split / "labels" / f"{label_basename}.txt"

                if image_path.exists():
                    shutil.copyfile(label_file, label_dest)
                    num_updated += 1
                    found = True
                    break

            if not found:
                not_found.append(label_file.name)

        print(f"✅ Updated {num_updated} labels.")
        if not_found:
            print(f"⚠️ {len(not_found)} labels had no matching image:")
            for nf in not_found:
                print(f"  - {nf}")


class DatasetSeparator:
    def __init__(self, dataset_dir, output_fully_labeled, output_pseudo_labeled, target_class_id="0", min_target_count=12):
        self.dataset_dir = Path(dataset_dir)
        self.output_fully_labeled = Path(output_fully_labeled)
        self.output_pseudo_labeled = Path(output_pseudo_labeled)
        self.target_class_id = target_class_id
        self.min_target_count = min_target_count

    def separate(self):
        """Separate the dataset into fully labeled and pseudo-labeled parts based on the target class id and min target count"""
        # Create output directories with images and labels subdirectories
        for output_dir in [self.output_fully_labeled, self.output_pseudo_labeled]:
            os.makedirs(output_dir / "images", exist_ok=True)
            os.makedirs(output_dir / "labels", exist_ok=True)

        # Get all image files
        image_files = [f for f in os.listdir(self.dataset_dir / "images") if f.endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            label_path = self.dataset_dir / "labels" / label_file

            if not label_path.exists():
                continue

            # Count target class instances
            target_count = 0
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip().split()[0] == self.target_class_id:
                        target_count += 1

            # Move files to appropriate directory
            if target_count >= self.min_target_count:
                shutil.move(
                    self.dataset_dir / "images" / img_file,
                    self.output_fully_labeled / "images" / img_file
                )
                shutil.move(
                    label_path,
                    self.output_fully_labeled / "labels" / label_file
                )
            else:
                shutil.move(
                    self.dataset_dir / "images" / img_file,
                    self.output_pseudo_labeled / "images" / img_file
                )
                shutil.move(
                    label_path,
                    self.output_pseudo_labeled / "labels" / label_file
                )

        # Try to delete the original directory after all files are moved
        try:
            shutil.rmtree(self.dataset_dir)
            print(f"✅ Dataset separated and original directory removed")
        except PermissionError:
            print(f"⚠️ Could not delete original directory due to permission error. You may need to delete it manually.")
        except Exception as e:
            print(f"⚠️ Could not delete original directory: {str(e)}")


class LabelStudioConverter:
    def __init__(self, type, split, data_dir, output_json, api_key="e728f34dbc49bc1809b6b2afe4e54609ffae1578", host="http://localhost:8080"):
        self.output_json = Path(output_json)
        self.data_dir = Path(data_dir)
        self.type = type
        self.split = split
        self.target_dir = Path(f"C:/datasets/{self.type}/{self.split}")
        self.class_names = {
            "goalkeeper": ["goalkeeper"],
            "player": ["player"],
            "ball": ["ball"]
        }
        self.project_manager = LabelStudioProjectManager(api_key=api_key, host=host)

    def create_classes_file(self):
        """Create classes.txt file in the target directory"""
        if self.type not in self.class_names:
            print(f"⚠️ Warning: No class names defined for type '{self.type}'")
            return False
            
        classes_file = self.target_dir / "classes.txt"
        with open(classes_file, "w") as f:
            f.write("\n".join(self.class_names[self.type]))
        print(f"✅ Created classes.txt with classes: {', '.join(self.class_names[self.type])}")
        return True

    def convert(self):
        """Convert YOLO format to Label Studio JSON format using label-studio-converter"""
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Copy the data directory to C:/datasets/type/
        try:
            # If the target directory already has content, remove it first
            if self.target_dir.exists():
                shutil.rmtree(self.target_dir)
            
            # Copy the entire data directory
            shutil.copytree(self.data_dir, self.target_dir, dirs_exist_ok=True)
            print(f"✅ Copied data to {self.target_dir}")
        except Exception as e:
            print(f"❌ Error copying data: {str(e)}")
            return

        # Create classes.txt file
        if not self.create_classes_file():
            print("❌ Failed to create classes.txt file")
            return

        # Set environment variables for the current process
        os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'
        os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = str(self.target_dir)

        # Construct the command
        cmd = [
            'label-studio-converter',
            'import',
            'yolo',
            '-i', str(self.target_dir),
            '-o', str(self.output_json),
            '--image-root-url', f'/data/local-files/?d={self.type}/{self.split}/images'
        ]

        try:
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully converted dataset to {self.output_json}")
                if result.stdout:
                    print(result.stdout)
                
                # Setup Label Studio project
                label_config_path = self.output_json.parent / f"{self.type}.label_config.xml"
                if label_config_path.exists():
                    print("🔄 Setting up Label Studio project...")
                    self.project_manager.setup_project(
                        title=f"{self.type.capitalize()} Dataset",
                        label_config_path=str(label_config_path),
                        storage_path=str(self.target_dir / "images"),
                        tasks_json_path=str(self.output_json)
                    )
                else:
                    print(f"⚠️ Label config file not found at {label_config_path}")
            else:
                print(f"❌ Error converting dataset:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error running label-studio-converter: {str(e)}")


class DatasetConsolidator:
    def __init__(self, input_dirs, output_dir):
        """
        Initialize the consolidator.
        
        Args:
            input_dirs (list): List of paths to the fully labeled datasets to merge
            output_dir (str): Path where the consolidated dataset will be created
        """
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.splits = ['train', 'valid', 'test']

    def consolidate(self):
        """Merge multiple fully labeled datasets into a single organized dataset"""
        # Create the main output directory structure
        for split in self.splits:
            os.makedirs(self.output_dir / split / "images", exist_ok=True)
            os.makedirs(self.output_dir / split / "labels", exist_ok=True)

        # Process each input directory
        for input_dir in self.input_dirs:
            # Get the split name from the directory name
            split_name = input_dir.name.split('_')[0]  # Assumes format like "train_fully_labeled"
            if split_name not in self.splits:
                print(f"⚠️ Skipping {input_dir} - not a valid split directory")
                continue

            # Move images
            images_dir = input_dir / "images"
            if images_dir.exists():
                for img_file in images_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.move(
                            str(img_file),
                            str(self.output_dir / split_name / "images" / img_file.name)
                        )

            # Move labels
            labels_dir = input_dir / "labels"
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    shutil.move(
                        str(label_file),
                        str(self.output_dir / split_name / "labels" / label_file.name)
                    )

            # Try to remove the original directory
            try:
                shutil.rmtree(input_dir)
                print(f"✅ Removed original directory: {input_dir}")
            except Exception as e:
                print(f"⚠️ Could not remove {input_dir}: {str(e)}")

        print(f"✅ Dataset consolidation complete. Output at: {self.output_dir}")


class LabelStudioProjectManager:
    def __init__(self, api_key, host="http://localhost:8080"):
        self.api_key = api_key
        self.host = host.rstrip('/')
        self.headers = {
            'Authorization': f'Token {api_key}',
            'Content-Type': 'application/json'
        }
        self.server_process = None

    def check_server_running(self):
        """Check if Label Studio server is running"""
        try:
            response = requests.get(f"{self.host}/api/health")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def create_project(self, title, label_config_path):
        """Create a new project with the specified label config"""
        if not self.check_server_running():
            print("❌ Label Studio server is not running")
            return None

        url = f"{self.host}/api/projects"
        
        # Read the label config file
        with open(label_config_path, 'r') as f:
            label_config = f.read()

        data = {
            "title": title,
            "label_config": label_config,
            "description": f"Project for {title} dataset"
        }

        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 201:
            project_id = response.json()['id']
            print(f"✅ Created project '{title}' with ID: {project_id}")
            return project_id
        else:
            print(f"❌ Failed to create project: {response.text}")
            return None

    def setup_local_storage(self, project_id, storage_path):
        """Setup local storage for the project"""
        if not self.check_server_running():
            print("❌ Label Studio server is not running. Starting server...")
            if not self.start_server():
                return False

        # First, ensure the storage path exists and is absolute
        storage_path = str(Path(storage_path).resolve())
        
        # Create storage configuration
        url = f"{self.host}/api/storages/localfiles"
        
        data = {
            "project": project_id,
            "path": storage_path,
            "regex_filter": ".*(jpg|jpeg|png)$",
            "use_blob_urls": True,
            "title": "Local Storage",
            "description": "Local storage for images",
            "presign": True,
            "presign_ttl": 1,
            "recursive_scan": True
        }

        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                storage_id = response.json()['id']
                print(f"✅ Setup local storage for project {project_id} with ID: {storage_id}")
                return True
            else:
                print(f"❌ Failed to setup storage: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error setting up storage: {str(e)}")
            return False

    def import_tasks(self, project_id, json_path):
        """Import tasks from a JSON file"""
        print("\n🔄 Starting task import process...")
        
        if not self.check_server_running():
            print("❌ Label Studio server is not running")
            return False

        url = f"{self.host}/api/projects/{project_id}/import"
        print(f"📡 Using import URL: {url}")
        
        # Read the JSON file
        try:
            print(f"📂 Reading tasks from {json_path}...")
            with open(json_path, 'r') as f:
                tasks = json.load(f)
            print(f"✅ Successfully loaded {len(tasks)} tasks from JSON file")
        except Exception as e:
            print(f"❌ Error reading JSON file: {str(e)}")
            return False

        # Prepare tasks in the correct format
        formatted_tasks = []
        for task in tasks:
            if isinstance(task, dict):
                # Get the image filename from the task
                image_filename = task.get("image", "")
                if not image_filename:
                    print(f"⚠️ Skipping task with missing image: {task}")
                    continue
                
                # Create the image URL using the local files serving path
                image_url = f"/data/local-files/?d={image_filename}"
                
                formatted_tasks.append({
                    "data": {
                        "image": image_filename,
                        "image_url": image_url
                    }
                })
            else:
                print(f"⚠️ Skipping invalid task format: {task}")

        data = {
            "tasks": formatted_tasks
        }

        try:
            print(f"🔄 Sending {len(formatted_tasks)} tasks to Label Studio...")
            response = requests.post(url, headers=self.headers, json=data)
            
            if response.status_code == 201:
                print(f"✅ Successfully imported {len(formatted_tasks)} tasks to project {project_id}")
                return True
            else:
                print(f"❌ Failed to import tasks. Status code: {response.status_code}")
                print(f"Error response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Error during task import: {str(e)}")
            return False

    def setup_project(self, title, label_config_path, storage_path, tasks_json_path):
        """Complete project setup in one go"""
        if not self.check_server_running():
            print("❌ Label Studio server is not running")
            return False

        print("\n🔄 Starting project creation and task import process...")
        
        # Create project
        print("\n1️⃣ Creating project...")
        project_id = self.create_project(title, label_config_path)
        if not project_id:
            print("❌ Project creation failed")
            return False

        # Setup storage
        print("\2️⃣ Setting up storage...")
        if not self.setup_local_storage(project_id, storage_path):
            print("❌ Storage setup failed")
            return False

        # Import tasks
        print("\n2️⃣ Importing tasks...")
        if not self.import_tasks(project_id, tasks_json_path):
            print("❌ Task import failed")
            return False

        print(f"\n✅ Project '{title}' created and tasks imported successfully!")
        return True


class DatasetWithYamlAsZip:
    def __init__(self, folder_path: str, class_name: str, yaml_filename: str = "data.yaml"):
        self.folder_path = Path(folder_path)
        self.class_name = class_name
        self.yaml_filename = yaml_filename
        self.zip_path = Path(f"data/kaggle_zips/{self.class_name}/{self.class_name}").with_suffix('.zip')

    def get_yaml_content(self) -> dict:
        return {
            "names": [self.class_name],
            "nc": 1,
            "test": "../test/images",
            "train": "../train/images",
            "val": "../valid/images"
        }

    def create_yaml_file(self, temp_dir: Path) -> Path:
        yaml_path = temp_dir / self.yaml_filename
        with open(yaml_path, 'w') as f:
            yaml.dump(self.get_yaml_content(), f, default_flow_style=False)
        return yaml_path

    def zip_folder_with_yaml(self):
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"{self.folder_path} is not a valid folder.")

        with zipfile.ZipFile(self.zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from the folder
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    abs_path = Path(root) / file
                    rel_path = abs_path.relative_to(self.folder_path.parent)
                    zipf.write(abs_path, arcname=rel_path)

            # Create and add the YAML file
            yaml_path = self.create_yaml_file(Path.cwd())
            zipf.write(yaml_path, arcname=self.folder_path.name + '/' + self.yaml_filename)
            os.remove(yaml_path)

        print(f"Created ZIP with YAML at: {self.zip_path}")


# Example usage:
if __name__ == "__main__":
    merger = DatasetMerger(
        parent_dataset_dir="player_training/player_dataset_v1",
        other_dataset_dir="datasets/waterpolo_ball_dataset",
        decider="remove"
    )
    merger.merge()

    
    updater = LabelUpdater(
        new_label_dir="datasets/label_studio/labels",
        dataset_dir="ball_training/datasets/waterpolo_ball_dataset/rejected"
    )
    updater.update()

    
    separator = DatasetSeparator(
        dataset_dir="filtered_dataset/valid",
        output_fully_labeled="datasets/valid_fully_labeled",
        output_pseudo_labeled="datasets/valid_pseudo_labeled",
        target_class_id="0",
        min_target_count=12
    )
    separator.separate()

    
    consolidator = DatasetConsolidator(
        input_dirs=[
            "datasets/train_fully_labeled",
            "datasets/valid_fully_labeled",
            "datasets/test_fully_labeled"
        ],
        output_dir="datasets/consolidated_dataset"
    )
    consolidator.consolidate()


    converter = LabelStudioConverter(
        type="goalkeeper",  
        data_dir="data/processed/goalkeeper/v1/train",
        output_json="data/label_studio/exports/goalkeeper_import.json"
    )
    converter.convert()

    
    project_manager = LabelStudioProjectManager(
        api_key="e728f34dbc49bc1809b6b2afe4e54609ffae1578",  
        host="http://localhost:8080"  
    )
    
    project_manager.setup_project(
        title="Goalkeeper Dataset",
        label_config_path="data/label_studio/exports/goalkeeper.label_config.xml",
        storage_path="C:/datasets/goalkeeper/images",  
        tasks_json_path="data/label_studio/exports/goalkeeper.json"
    )


    merger = DatasetMerger(
        parent_dataset_dir="player_training/player_dataset_v1",
        other_dataset_dir="datasets/waterpolo_ball_dataset",
        decider="copy"
    )
    merger.merge()


    zipper = DatasetWithYamlAsZip(
         folder_path="my_dataset_folder",
        class_name="goalkeeper"
    )
    zipper.zip_folder_with_yaml()