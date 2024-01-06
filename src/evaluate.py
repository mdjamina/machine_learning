# -*- coding: utf-8 -*-
import argparse
import re
from pathlib import Path

import joblib
import spacy
from lxml import etree as et
from pandas import read_csv, read_pickle, DataFrame
from sklearn import metrics
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
        u'\0xa0': ' ',

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
        text = " ".join(p.text for p in doc.findall('.//p') if p.text)

        yield id_doc, text


def load_label(datafile: str):
    """Charger le fichier de référence
    format csv (index, label)

    Args:
        datafile (str): fichier de référence

    Returns:
        data (pd.DataFrame): données chargées

    """

    data = read_csv(datafile, sep='\t', header=None, names=['index', 'label']).head(500)
    data = data.set_index('index')
    return data.label.to_numpy()


def nlp_load():
    """Charger le modèle de langue et désactiver les pipelines inutiles

    returns:
        nlp: modèle de langue chargé

    """
    disable = ["vectors", "senter", "textcat"]

    return spacy.load('fr_core_news_lg', disable=disable)


def nlp_preprocess(text: str) -> str:
    """Prétraiter le texte

    Args:
        text (str): texte à prétraiter

    Returns:
        str: texte prétraité
    """

    doc = nlp(text)

    document = []

    for token in doc:
        is_stop = any([token.lemma_ in STOP_WORDS,
                       token.text in STOP_WORDS,
                       token.is_stop,
                       token.is_punct,
                       token.is_space,
                       token.is_digit
                       ])
        if not is_stop:
            document.append(token.lemma_.lower())

    return " ".join(document)


def load_test(pathfile: str, cleaning: bool = True):
    """Charger le fichier de test
    format csv (index, label)

    Args:
        cleaning:
        pathfile:


    Returns:
        data (pd.DataFrame): données chargées

    """
    global nlp
    nlp = nlp_load()

    pathfile = Path(pathfile)
    if not pathfile.exists():
        raise FileNotFoundError(f"{pathfile} not found")

    # chargement à partir d'un fichier xml
    if pathfile.suffix == '.xml':
        data = DataFrame(parser(str(pathfile), cleaning=cleaning), columns=['index', 'docs']).head(500)
        data['docs'] = data.docs.apply(lambda x: nlp_preprocess(x))
        data = data.set_index('index')
        # save data
        data.to_pickle(Path(pathfile).parent / f"{Path(pathfile).stem}.pkl")

    elif pathfile.suffix == '.pkl':
        data = read_pickle(pathfile)


    else:
        raise ValueError(f"file format {pathfile.suffix} not supported")

    return data.docs.to_numpy()


def load_model(pathfile: str):
    """Charger le modèle

    Args:
        pathfile (str): chemin du fichier

    Returns:
        model: modèle chargé
    """
    pathfile = Path(pathfile)
    if not pathfile.exists():
        raise FileNotFoundError(f"{pathfile} not found")

    return joblib.load(pathfile)


def main():
    """Évaluer le modèle
    --model ./models/model.pkl  --datatest ./data/deft09_parlement_appr_fr.xml --dataref ./data/deft09_parlement_appr_fr.xml
    """

    arg = argparse.ArgumentParser()
    arg.add_argument('--model', type=str, required=True, help='model path')
    arg.add_argument('--datatest', type=str, required=True, help='test data path')
    arg.add_argument('--dataref', type=str, required=True, help='reference data path')
    arg.add_argument('--cleaning', type=bool, default=True, help='cleaning data')

    args = arg.parse_args()

    # model directory
    model = args.model

    # data directory
    datatest = args.datatest
    dataref = args.dataref
    cleaning = args.cleaning

    # load data
    x_test = load_test(datatest, cleaning=cleaning)
    y_test = load_label(dataref)

    # load model
    model = load_model('model.pkl')

    # predict
    predictions = model.predict(x_test)

    # print results
    print(metrics.classification_report(y_test, predictions))
    print(metrics.confusion_matrix(y_test, predictions))
    print(model.score(x_test, y_test))

    exit(0)


if __name__ == '__main__':
    main()
