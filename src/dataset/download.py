from datasets import load_dataset


def downloadKomeijiForce(path):
    dataset = load_dataset('KomeijiForce/Text2Emoji', split='train')
    dataset.save_to_disk(path)
    return dataset
