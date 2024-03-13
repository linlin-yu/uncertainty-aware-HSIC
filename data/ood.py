from typing import Optional
import math
import torch
import numpy as np


# partially adapted from https://github.com/stadlmax/Graph-Posterior-Network
def get_ood_split(
    data,
    ood_frac_left_out_classes: float = 0.45,
    ood_num_left_out_classes: Optional[int] = None,
    ood_leave_out_last_classes: Optional[bool] = False,
    ood_left_out_classes: Optional[list] = None,
    **_
):
    """splits data in ID and OOD data for Leave-Out-Classes experiment
    The split can be either specified by specifying the fraction of left-out classes, the number of left-out-classes, or by passing a list of class
    indices to leave out. In the first two cases, the flag ood_leave_out_last_classes can be set to leave the last class indices out. Otherwise,
    the left-out classes are simply sampled randomly.
    Args:
        data (torch_geometric.data.Data): data object representing graph data
        ood_frac_left_out_classes (float, optional): fraction nof left-out classes. Defaults to 0.45.
        ood_num_left_out_classes (Optional[int], optional): number of left-out classes. Defaults to None.
        ood_leave_out_last_classes (Optional[bool], optional): whether or whether not to leave the last class indices out (assuming c in [1, ... C]). Defaults to False.
        ood_left_out_classes (Optional[list], optional): optional list of class indices to leave out. Defaults to None.
    Returns:
        Tuple[torch_geometric.data.Data, int]: tuple of data object and number of classes in the ID case
    """

    # creates masks / data copies for ood dataset (left out classes) and
    # default dataset (without the classes being left out)
    data = data.clone()

    assert (
        hasattr(data, "train_mask")
        and hasattr(data, "val_mask")
        and hasattr(data, "test_mask")
    )

    num_classes = data.y.max().item() + 1
    classes = np.arange(num_classes)

    if ood_left_out_classes is None:
        # which classes are left out
        if ood_num_left_out_classes is None:
            ood_num_left_out_classes = math.floor(
                num_classes * ood_frac_left_out_classes
            )

        if not ood_leave_out_last_classes:
            # create random perturbation of classes to leave out
            # classes in the end
            # if not specified: leave out classes which are originally
            # at the end of the array of sorted classes
            np.random.shuffle(classes)

        left_out_classes = classes[num_classes - ood_num_left_out_classes : num_classes]
        id_classes = [c for c in classes if c not in left_out_classes]

    else:
        ood_num_left_out_classes = len(ood_left_out_classes)
        left_out_classes = np.array(ood_left_out_classes)
        # reorder c in classes, such that left-out-classes
        # are at the end of classes-array
        id_classes = [c for c in classes if c not in left_out_classes]
        tmp = id_classes + [c for c in classes if c in left_out_classes]
        classes = np.array(tmp)
    data.id_classes = id_classes

    # key is the original class index and i is the new class index
    class_mapping = {c: i for i, c in enumerate(classes)}
    data.class_mapping = class_mapping

    # ood labeled mask
    left_out = torch.zeros_like(data.y, dtype=bool)
    for c in left_out_classes:
        left_out = left_out | (data.y == c)

    # data.labeled_mask = data.train_mask | data.val_mask | data.test_mask
    data.ood_mask = left_out

    data.ood_train_mask = left_out & data.train_mask
    data.ood_val_mask = left_out & data.val_mask
    data.ood_test_mask = left_out & data.test_mask

    data.train_mask[left_out] = False
    data.test_mask[left_out] = False
    data.val_mask[left_out] = False

    data.id_train_mask = data.train_mask
    data.id_val_mask = data.val_mask
    data.id_test_mask = data.test_mask

    data.id_mask = data.train_mask | data.val_mask | data.test_mask

    num_classes = num_classes - ood_num_left_out_classes

    # finally apply positional mapping of classes from above to ensure that
    # classes are ordered properly (i.e. create dataset with labels being in range 0 --- new_num_classes - 1)
    data.y = torch.LongTensor(
        [class_mapping[y.item()] for y in data.y], device=data.y.device
    )
    # print(data.endmemberS.shape)
    # print(left_out_classes)
    # print(id_classes)
    # input("")
    if hasattr(data, "endmemberS"):
        data.endmemberS_00 = data.endmemberS[left_out_classes, :]
        data.endmemberS = data.endmemberS[id_classes, :]
    # print(data.endmemberS.shape)
    # print(data.endmemberS_00.shape)
    # input("")
    return data, num_classes


# if __name__ == '__main__':
#     data = DatasetManager('PaviaU', './data', split='consistent',train_samples_per_class=50, val_size = 0.2, test_size=0.8)
#     data, num_classes = get_ood_split(data[0], ood_left_out_classes=[4] )
# idx = data.id_val_mask.nonzero(as_tuple=False).view(-1)
# # print(idx)
# print(data.y.size())
# print(set(np.array(data.y[idx])))
# print(num_classes)
