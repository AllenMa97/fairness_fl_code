import numpy as np


def client_selection(client_num, fraction, dataset_size, client_dataset_size_list, drop_rate, probabilities=None, style="FedAvg"):
    if probabilities is None:
        probabilities = []
    assert sum(client_dataset_size_list) == dataset_size
    idxs_users = [0]

    selected_num = max(int(fraction * client_num), 1)
    if float(drop_rate) != 0:
        drop_num = max(int(selected_num * drop_rate), 1)
        selected_num -= drop_num

    if "FedAvg".lower() in style.lower():
        idxs_users = np.random.choice(
            a=range(client_num),
            size=selected_num,
            replace=False,
        )

    elif "FedProx".lower() in style.lower():
        values = [(i/dataset_size) for i in client_dataset_size_list]
        probabilities = np.array(values)
        idxs_users = np.random.choice(
            a=range(client_num),
            size=selected_num,
            replace=False,
            p=probabilities
        )

    elif "test".lower() in style.lower():
        values = [0.2] * 4 + [0.2 / (client_num - 4)] * (client_num - 4)
        probabilities = np.array(values)
        idxs_users = np.random.choice(
            a=range(client_num),
            size=selected_num,
            replace=False,
            p=probabilities
        )
    elif "giant".lower() in style.lower():
        if probabilities==None:
            raise AssertionError("algorithm fed giant should transmit probabilities(related to rank)")
        idxs_users = np.random.choice(
            a=range(client_num),
            size=selected_num,
            replace=False,
            p=probabilities
        )


    return idxs_users