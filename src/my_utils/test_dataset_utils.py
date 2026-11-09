from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import Dataset

from src.my_utils.data_utils import TASKS, Sample, discover, load_input, validate_samples

ROOT = Path(__file__).resolve().parents[2]


class IRTestDataset(Dataset):
    def __init__(self, dataset_opt, args=None):
        if args is None:
            args = SimpleNamespace(
                dataset_config={"datasets": {"val": dataset_opt}},
                group_file=None,
                input=None,
                all_in_one=False,
                output_dir=Path(dataset_opt.get("output_dir", ROOT / "outputs")),
                tasks=dataset_opt.get("de_type", list(TASKS)),
                limit=dataset_opt.get("limit", 0),
                dataset_root=Path(
                    dataset_opt.get("dataroot", "/path/to/datasets/all_in_one/val")
                ),
                gt_dir=None,
            )
        group_files = args.group_file or (
            None if args.input or args.all_in_one else dataset_opt.get("group_file")
        )
        if group_files:
            self.samples = []
            for group_file in group_files:
                path = Path(group_file)
                if not path.is_absolute():
                    path = ROOT / path
                per_task = {}
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    lq, gt = (Path(part.strip()) for part in line.split("|"))
                    if not lq.is_file() or not gt.is_file():
                        raise FileNotFoundError(f"Missing test pair: {lq} | {gt}")
                    task = next(
                        (task for task, folder in TASKS.items() if folder in lq.parts),
                        "custom",
                    )
                    if task != "custom" and task not in args.tasks:
                        continue
                    per_task[task] = per_task.get(task, 0) + 1
                    if args.limit and per_task[task] > args.limit:
                        continue
                    parent = next(
                        (p for p in lq.parents if p.name.lower() == "lq"), lq.parent
                    )
                    self.samples.append(
                        Sample(
                            lq,
                            args.output_dir
                            / task
                            / lq.relative_to(parent).with_suffix(".png"),
                            task,
                            gt,
                        )
                    )
            self.samples = validate_samples(self.samples)
        else:
            self.samples = discover(
                args.input,
                args.dataset_root,
                args.output_dir,
                args.tasks,
                args.limit,
                args.gt_dir,
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        lq, (height, width) = load_input(sample.source)
        result = {
            "lq": lq[0],
            "lq_path": str(sample.source),
            "degra": sample.task,
            "ori_h": height,
            "ori_w": width,
        }
        if sample.reference:
            gt, _ = load_input(sample.reference)
            result.update(gt=gt[0], gt_path=str(sample.reference))
        return result
