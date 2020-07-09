"""Extracts the text from a .pdf document.

This file is adapted from research by:

@article{bakarovrussian,
  title={Russian Computational Linguistics: Topical Structure in 2007-2017 Conference Papers},
  journal={Komp'yuternaya Lingvistika i Intellektual'nye Tekhnologii},
  year={2018},
  author={Bakarov, Amir and Kutuzov, Andrey and Nikishina, Irina}
}

source: https://github.com/rusnlp/rusnlp
"""

from os import path, walk, makedirs
from io import StringIO
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage


def convert_pdf_to_txt(filepath, page_range = (0,0)):
    rm = PDFResourceManager()
    sio = StringIO()
    device = TextConverter(rm, sio, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rm, device)
    with open(filepath, 'rb') as fp:
        for page in PDFPage.get_pages(fp=fp, pagenos=set(), maxpages=page_range[1], password='',
                                      caching=True, check_extractable=True)[page_range[0]]:
            interpreter.process_page(page)
    text = sio.getvalue()
    device.close()
    sio.close()
    return text


def write_errors(errors):
    with open('errors.txt', 'w', encoding='utf-8') as f:
        for error in errors:
            f.write('{}\n'.format(error))


def write_file(source_dir, saving_dir, root, file, text):
    with open(path.join(root.replace(source_dir, saving_dir), file).replace('.pdf', '.txt'), 'w',
              encoding='utf-8') as f:
        f.write(text)


def convert(source_dir, saving_dir, page_range = (0,0)):
    errors = []
    for root, dirs, files in walk(source_dir):
        for file in files:
            if not file.endswith('pdf'):
                errors.append(path.join(root, file))
                continue
            text = convert_pdf_to_txt(path.join(root, file), page_range)
            try:
                write_file(source_dir, saving_dir, root, file, text)
            except FileNotFoundError:
                makedirs(path.dirname(path.join(root.replace(source_dir, saving_dir), file)))
                write_file(source_dir, saving_dir, root, file, text)
    write_errors(errors)


if __name__ == '__main__':
    convert(source_dir='parsed', saving_dir='sources')