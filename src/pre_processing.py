"""Ce script effectue les étapes nécessaires pour prétraiter le corpus, y compris la tokenisation, la suppression des
stop words et la lemmatisation. Contribue à une meilleure qualité des données d'entraînement.
"""
import argparse
import re
import timeit as ti
from pathlib import Path
import pandas as pd
import spacy
from lxml import etree as et
from spacy.lang.fr.stop_words import STOP_WORDS
from tqdm import tqdm


def replace_characters(match: re.Match) -> str:
    """Remplacer les caractères spéciaux

    Args:
        match : caractère à remplacer

    Returns:
        str: caractère remplacé
    """
    char = match.group(0)
    replacements = {
        '’': "'",
        '´': "'",
        '`': "'",
        '‘': "'",
        '«': '"',
        '»': '"',
        '“': '"',
        '”': '"',
        '–': '-',
        '—': '-',
        '…': ' ',
        u'\xa0': ' ',
    }

    return replacements[char]


def normalize_text(text: str) -> str:
    """Normaliser le texte

    Args:
        text (str): texte à normaliser

    Returns:
        str: texte normalisé
    """

    pattern = r'[’´`‘«»“”–—…]'

    return re.sub(pattern, replace_characters, text).strip()


def fix_file_source(pathfile: str) -> Path:
    """Corriger le fichier source

    Args:
        pathfile (str): chemin du fichier xml

    Returns:
        Path: chemin du fichier xml corrigé
    """

    pathfile = Path(pathfile)
    if not pathfile.exists():
        raise FileNotFoundError(f"{pathfile} not found")

    pathfile_fixed = pathfile

    pathfile_backup = pathfile.parent / f"{pathfile.stem}_backup{pathfile.suffix}"
    pathfile.rename(pathfile_backup)

    # pathfile_fixed = pathfile.parent / f"{pathfile.stem}_fixed{pathfile.suffix}"

    with pathfile_backup.open('r') as f:
        # remplacement de <anonyme /> par ANONYME
        pattern = re.compile(r'<anonyme\s*/>')
        content = re.sub(pattern, "ANONYME", f.read())
        content = normalize_text(content)
    with pathfile_fixed.open('w') as f2:
        f2.write(content)

    return pathfile_fixed


def parser(pathfile: str, cleaning: bool = True) -> list:
    """Parser le fichier xml

    Args:
        pathfile (str): chemin du fichier xml
        cleaning (bool, optional): nettoyer le fichier xml. Defaults to False.

    Yields:
        list: liste des documents
    """

    if cleaning:
        pathfile = fix_file_source(pathfile)

    # Parser le fichier XML avec lxml pour plus de performance
    tree = et.parse(pathfile)

    # Récupérer la racine du fichier XML
    root = tree.getroot()

    # Utiliser tqdm pour afficher la barre de progression
    for doc in tqdm(root.findall('.//doc'), desc="Parsing XML"):
        id_doc: str = doc.attrib['id']
        parti = doc.find('.//PARTI').attrib['valeur']
        text = " ".join(p.text for p in doc.findall('.//p') if p.text)

        yield id_doc, parti, text


def load_corpus(pathfile: str, cleaning: bool = True, size: int = None, random_state: int = None) -> pd.DataFrame:
    """Charger le corpus à partir d'un fichier parquet ou xml

    Args:
        cleaning:
        random_state: graine aléatoire  pour la sélection des données
        size: taille du corpus à traiter
        pathfile (str): chemin du fichier


    Returns:
        pd.DataFrame: corpus
    """

    pathfile = Path(pathfile)
    if not pathfile.exists():
        raise FileNotFoundError(f"{pathfile} not found")

    # chargement à partir d'un fichier xml
    if pathfile.suffix == '.xml':
        data = pd.DataFrame(parser(str(pathfile), cleaning=cleaning), columns=['id', 'parti', 'text'])

    else:
        raise ValueError(f"file format {pathfile.suffix} not supported")
    # échantillonage du corpus
    if size:
        # selectionner même nombre de données de chaque parti de manière aléatoire
        data = data.groupby('parti').apply(lambda x: x.sample(size // len(data.parti.unique())
                                                              , random_state=random_state)).reset_index(drop=True)
    return data


def nlp_load():
    """Charger le modèle de langue et désactiver les pipelines inutiles

    returns:
        nlp: modèle de langue chargé

    """
    disable = ["vectors", "senter", "textcat"]

    return spacy.load('fr_core_news_lg', disable=disable)


def nlp_preprocess(text: str) -> list:
    """Prétraiter le texte

    Args:
        text (str): texte à prétraiter

    Returns:
        tuple: texte prétraité
    """

    doc = nlp(text)

    # ENR : entités nommées
    #entities = [(ent.text, ent.label_) for ent in doc.ents]

    document = []

    for token in doc:
        is_stop = any([token.lemma_ in STOP_WORDS,
                       token.text in STOP_WORDS,
                       token.is_stop,
                       token.is_punct,
                       token.is_space,
                       token.is_digit
                       ])

        document.append(
            {'form': token.text, 'lemma': token.lemma_.lower(), 'pos': token.pos_,
             'is_stop': is_stop})

    #return document, entities
    return document


def preprocess_corpus(pathfile: str, size: int = None, random_state: int = 85, cleaning: bool = False) -> Path:
    """Prétraiter le corpus

    Args:
        random_state:
        size: taille du corpus à traiter
        pathfile (str): chemin du fichier corpus

    Returns:
        Path: chemin du fichier corpus prétraité (joblib)

    """

    print("chargement du corpus")

    pathfile_pickle = Path(pathfile).parent / f"{Path(pathfile).stem}_pre.dat"

    # chargement du corpus
    corpus = load_corpus(pathfile=pathfile, size=size, random_state=random_state, cleaning=cleaning)

    print("chargement du modèle spacy")
    start = ti.default_timer()
    global nlp
    nlp = nlp_load()
    stop = ti.default_timer()
    print('le temps chargement du modèle spacy: ', stop - start)

    # prétraitement du corpus

    print("prétraitement du corpus")
    start0 = ti.default_timer()

    #corpus['docs'], corpus['entities'] = zip(*corpus['text'].map(nlp_preprocess))

    corpus['docs'] = corpus['text'].map(nlp_preprocess)
    stop0 = ti.default_timer()
    print('le temps de prétraitement du corpus: ', stop0 - start0)

    # sauvegarde du corpus prétraité
    print("sauvegarde du corpus prétraité")

    corpus.to_pickle(pathfile_pickle)

    print("corpus prétraité dans: ", pathfile_pickle)

    print(corpus.head(100))

    return pathfile_pickle


"""
def anonymize(text: str, ent: list) -> str:
    ""Anonymiser les entités nommées
    ""

    print("ent",ent)

    for e in ent:
        text = re.sub(e[0], e[1], text)

    print("text", text)

    return text
 tmp['text'] = tmp.apply(lambda x: anonymize(x.text, x.entities), axis=1)
"""


def main():
    """Exécuter le script
    Args

    """
    # data directory
    arg = argparse.ArgumentParser()
    arg.add_argument('--datafile', type=str, required=True, help='chemin du fichier de données')
    arg.add_argument('--size', type=int, required=False, help='taille du corpus à traiter'
                     , default=None)
    arg.add_argument('--random_state', type=int, required=False, help='graine aléatoire pour la sélection des données'
                     , default=85)
    arg.add_argument('--cleaning', type=bool, required=False, help='nettoyer le fichier xml'
                     , default=False)

    args = arg.parse_args()

    datafile = args.datafile
    size = args.size
    random_state = args.random_state
    cleaning = args.cleaning
    #

    # initialisation du modèle

    # load data

    output = preprocess_corpus(pathfile=datafile, size=size, random_state=random_state, cleaning=cleaning)

    print("fichier dans: ", output)


if __name__ == "__main__":
    main()
