from nltk import download


def download_nltk_req() -> None:
    """
    Download nltk requirements for class Text2EmojiParser.
    :return: None
    """
    download("stopwords")
    download("punkt")
    download("wordnet")
