
"""Ce script effectue les étapes nécessaires pour prétraiter le corpus, y compris la tokenisation, la suppression des
stop words et la lemmatisation. Contribue à une meilleure qualité des données d'entraînement.
"""


import csv
import re
import pandas as pd
from pathlib import Path
from lxml import etree as et
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
import timeit as ti
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
        '…': '...',
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


def parser(pathfile: str, cleanfile: bool = True) -> list:
    """Parser le fichier xml

    Args:
        pathfile (str): chemin du fichier xml
        cleanfile (bool, optional): nettoyer le fichier xml. Defaults to False.

    Yields:
        list: liste des documents
    """

    if cleanfile:
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


def export_to_csv(pathfile: str, delimiter: str = ';', quotechar='"', quoting=csv.QUOTE_MINIMAL) -> Path:
    """générer un fichier csv à partir d'un fichier xml
    en utilsant la fonction func pour parser le fichier xml

    Args:
        pathfile (str): chemin du fichier xml
        delimiter (str, optional): délimiteur. Defaults to ';'.
        quotechar (str, optional): caractère de citation. Defaults to '"'.
        quoting ([type], optional): type de citation. Defaults to csv.QUOTE_MINIMAL.

    Returns:
        str: chemin du fichier csv généré
    """
    pathfile = Path(pathfile)
    path_csv = pathfile.parent / f"{pathfile.stem}.csv"

    with path_csv.open('w') as f:
        csvfile = csv.writer(f, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
        csvfile.writerow(['id', 'parti', 'text'])
        for id_doc, parti, text in parser(str(pathfile)):
            csvfile.writerow([id_doc, parti, text])

    return path_csv


def export_to_parquet(pathfile: str, cleanfile: bool = True) -> Path:
    """générer un fichier parquet à partir d'un fichier xml
    en utilsant la fonction func pour parser le fichier xml

    Args:
        pathfile (str): chemin du fichier xml
        cleanfile (bool, optional): nettoyer le fichier xml. Defaults to True.
    Returns:
        str: chemin du fichier parquet généré
    """

    pathfile = Path(pathfile)
    path_parquet = pathfile.parent / f"{pathfile.stem}.parquet"

    data = []
    for id_doc, parti, text in parser(str(pathfile), cleanfile=cleanfile):
        data.append([id_doc, parti, text])

    pd.DataFrame(data, columns=['id', 'parti', 'text']).to_parquet(path_parquet)

    return path_parquet


def load_corpus(pathfile: str, clean: bool = True) -> pd.DataFrame:
    """Charger le corpus à partir d'un fichier parquet ou xml

    Args:
        pathfile (str): chemin du fichier parquet
        clean (bool, optional): nettoyer le fichier xml. Defaults to False.

    Returns:
        pd.DataFrame: corpus
    """

    pathfile = Path(pathfile)
    if not pathfile.exists():
        raise FileNotFoundError(f"{pathfile} not found")

    # chargement à partir d'un fichier xml
    if pathfile.suffix == '.xml':
        data = pd.DataFrame(parser(str(pathfile), cleanfile=clean), columns=['id', 'parti', 'text'])

    # chargement à partir d'un fichier parquet
    elif pathfile.suffix == '.parquet':
        data = pd.read_parquet(pathfile)

    else:
        raise ValueError(f"file format {pathfile.suffix} not supported")

    return data


def nlp_load():
    """Charger le modèle de langue et désactiver les pipelines inutiles

    returns:
        nlp: modèle de langue chargé

    """
    disable = ["vectors", "senter", "textcat"]

    return spacy.load('fr_core_news_lg', disable=disable)


# initialisation du modèle
print("chargement du modèle spacy")
start = ti.default_timer()
nlp = nlp_load()
stop = ti.default_timer()
print('le temps chargement du modèle: ', stop - start)


def nlp_preprocess(text: str) -> tuple:
    """Prétraiter le texte

    Args:
        text (str): texte à prétraiter

    Returns:
        tuple: texte prétraité
    """

    doc = nlp(text)

    # ENR : entités nommées
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    document = []

    for token in doc:
        is_stop = any([token.lemma_ in STOP_WORDS,
                       token.text in STOP_WORDS,
                       token.is_stop,
                       token.is_punct,
                       token.is_space,
                       token.is_digit
                       ])

        document.append({'form': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'is_stop': is_stop})

    return document, entities


def pipline_preprocess(text: list) -> tuple:
    """Prétraiter le texte

    Args:
        text (list): texte à prétraiter

    Returns:
        tuple: texte prétraité
    """

    for doc in nlp.pipe(text, batch_size=1000, n_process=4):
        # print("doc: ", type(doc))
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        docuement = []
        for token in doc:
            is_stop = any([token.lemma_ in STOP_WORDS,
                           token.text in STOP_WORDS,
                           token.is_stop,
                           token.is_punct,
                           token.is_space,
                           token.is_digit
                           ])

            docuement.append({'form': token.text, 'lemma': token.lemma_, 'pos': token.pos_, 'is_stop': is_stop})

        yield docuement, entities


def preprocess_corpus(pathfile: str) -> Path:
    """Prétraiter le corpus

    Args:
        pathfile (str): chemin du fichier corpus

    Returns:
        Path: chemin du fichier corpus prétraité (joblib)

    """

    print("chargement du corpus")

    pathfile_pickle = Path(pathfile).parent / f"{Path(pathfile).stem}_preprocessed.dat"
    pthfile_parquet = Path(pathfile).parent / f"{Path(pathfile).stem}_preprocessed.parquet"

    # chargement du corpus
    corpus = load_corpus(path_file)

    # prétraitement du corpus

    print("prétraitement du corpus")
    start0 = ti.default_timer()

    corpus['doc'], corpus['entities'] = zip(*corpus['text'].map(nlp_preprocess))
    stop0 = ti.default_timer()
    print('le temps de prétraitement du corpus: ', stop0 - start0)

    # sauvegarde du corpus prétraité
    print("sauvegarde du corpus prétraité")

    corpus.to_pickle(pathfile_pickle)
    corpus.to_parquet(pthfile_parquet)

    return pathfile_pickle


if __name__ == "__main__":
    # test de la load_corpus
    path_file = "/home/amina/workspace/github/apprentissage-artificiel/data/deft09_parlement_appr_fr_dev.xml"

    print("chargement du corpus")
    print("fixer le fichier source")
    start1 = ti.default_timer()

    path_file = export_to_parquet(path_file)

    filename = preprocess_corpus(str(path_file))

    print("fichier dans: ", filename)
