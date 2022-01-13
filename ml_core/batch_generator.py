from data.data_loader import load_data


def batch_generator(data_path, index_list, batch_size=1):
    while True:
        batches = [index_list[i:i + batch_size] for i in [*range(0, len(index_list), batch_size)]]
        for batch in batches:
            images = load_data(data_path, index_list=batch)
            yield images['data'], images['label']
