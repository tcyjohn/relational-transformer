from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names

if __name__ == "__main__":
    for dataset_name in get_dataset_names():
        # 跳过 mimic 数据集
        if dataset_name == "rel-mimic":
            print("Skipping rel-mimic (requires medical credentials & Google Cloud)")
            continue
            
        print(f"Downloading dataset: {dataset_name}")
        get_dataset(dataset_name, download=True)

        for task_name in get_task_names(dataset_name):
            print(f"Downloading task: {task_name} from dataset: {dataset_name}")
            get_task(dataset_name, task_name, download=True)
