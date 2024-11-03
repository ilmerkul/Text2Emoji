import datasets


def downloadKomeijiForce(path: str) -> datasets.Dataset:
    """
    Load and save on disk KomeijiForce huggingface dataset.
    :param path: path for save on disk
    :return: KomeijiForce huggingface dataset
    """
    dataset = datasets.load_dataset('KomeijiForce/Text2Emoji', split='train')
    dataset.save_to_disk(path)
    return dataset
