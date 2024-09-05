import os
import re
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from math import log10, floor
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from matplotlib.backends.backend_pdf import PdfPages

###########################################################################
### Decoradores
###########################################################################
def check_data(func):
    def wrapper(data, *args, **kwargs):
        if isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a list or a numpy array")
        return func(data, *args, **kwargs)
    return wrapper
### ***

def round_to(num,n):
    '''
    Rounding a number to n significant digits
    E.g.(1): round_to(123456.789,1) -> 100000.0
    E.g.(2): round_to(123456.789,6) -> 123457.0
    E.g.(3): round_to(123456.789,8) -> 123456.79
    '''
    return round(num, n - int(floor(log10(abs(num)))) - 1)

def sort_list_of_fs_by_ascending_number(list_of_fs, r_pattern = ''):
    '''
    This function sorts a list of files/words by ascending order 
    E.g.(1): sort_list_of_fs_by_ascending_number(['Model_10.inp','Model_1.inp'])
    E.g.(2): sort_list_of_fs_by_ascending_number(['t_1 Model_10.inp','t_2 Model_1.inp'], 'Model_')
    ----------
    list_of_fs: [list of files/words]
    r_pattern: [str/regex] | regex pattern | Def. arg.: ''
    ----------
    the function modifies the list
    '''
    list_of_fs.sort(key=lambda el:int(re.search(f'{r_pattern}(\d+)',el).group(1)))

def save_images_to_pdf(pdf_name, images_folder = '.' ,  r_pattern = 'png', remove_imgs = False):
    image_files = [f for f in os.listdir(images_folder) if re.search(r_pattern, f)]
    sort_list_of_fs_by_ascending_number(image_files)

    pdf = canvas.Canvas(f'{images_folder}/{pdf_name}.pdf', pagesize=letter)

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file) # Sacar directorio de cada imagen
        
        with Image.open(image_path) as img:
            img_width, img_height = img.size
            
            # Calculate aspect ratio to fit within the page size
            aspect_ratio = img_width / img_height
            target_width = letter[0]
            target_height = target_width / aspect_ratio

            # Ensure the image fits within the page size
            if target_height > letter[1]:
                target_height = letter[1]
                target_width = target_height * aspect_ratio

            # Calculate centering position
            x_offset = (letter[0] - target_width) / 2
            y_offset = (letter[1] - target_height) / 2
            
            # Draw the image on the PDF with correct size and position
            pdf.drawImage(image_path, x_offset, y_offset, width=target_width, height=target_height)
            pdf.showPage() 
            
    pdf.save()
    if remove_imgs:[os.remove(f'{images_folder}/{f}') for f in image_files]


def create_temp_imgs(f_list, path = None):
    if not path: path = Path('.')
    [fig.savefig(f'{path}/{i+1}_temp_img.png') for i, fig in enumerate(f_list)]


def create_zip_given_f_l(zip_name, path = None,  r_pattern = 'png', remove_fs = False):
    '''
    This function generates a zip file given a folder and r_pattern.
    E.g.: create_zip_given_f_l('zip_file', path = Path('Results'),  r_pattern = 'png', remove_fs = True)            
    ----------
    zip_name: [str] | Name of the zip file
    path: [str] | path
    r_pattern: [str/regex] | regex pattern | Def. arg.: ''
    remove_fs: [True/False] | delete/don't delete  files  | Def. arg.: False
    '''
    # if not path: path = Path('.')
    if not path: path = '.'
    files = [f for f in os.listdir(path) if re.search(r_pattern, f)]
    # sort_list_of_fs_by_ascending_number(files)
    
    with zipfile.ZipFile(f'{path}/{zip_name}.zip', 'w') as img_zip:
        for f in files:
            img_zip.write(f'{path}/{f}', f)
    if remove_fs:[os.remove(f'{path}/{f}') for f in files]

def save_figs_to_pdf(path, f_list): 
    '''
    This function generates a pdf with a given figure list.
    E.g.: save_figs_as_pdf('Imagenes/pdf_doc.pdf', f_list)
    ----------
    path: [str] | Name of the pdf to be saved with | Def. arg.:  'pdf_doc.pdf'
    f_list: [list of figures] | [f1, f2, ..., fn]
    '''
    with PdfPages(path) as pdf:
        for f in f_list:
            pdf.savefig(f)

def get_folder_size(folder_path, unit = 'MB', include_subfolders=True):
    '''
    Get the folder size
    E.g.: get_folder_size(folder_path, unit = 'GB', include_subfolders=False)
    ----------
    folder_path: [df]
    unit: [list]
    include_subfolders: 
    '''
    unit_multipliers = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024, 'GB': 1024 * 1024 * 1024}
    if unit not in unit_multipliers:
        raise ValueError("Unidad no válida. Usa 'B', 'KB', 'MB' o 'GB'.")
    total_size = 0
    if include_subfolders:
        for dirpath, _, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    else:
        for f in os.listdir(folder_path):
            fp = os.path.join(folder_path, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    size_in_unit = total_size / unit_multipliers[unit]
    print(f'Tamaño de la carpeta: {size_in_unit:.2f} {unit}')
    return total_size

@check_data
def remove_outliers_IQR_meth(data):
    # IQR (Interquartile Range) Method
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    l_bound = Q1 - 1.5 * IQR
    u_bound = Q3 + 1.5 * IQR
    return data[(data >= l_bound) & (data <= u_bound)]

@check_data
def remove_outliers_zscore(data, threshold=3):
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return data[z_scores < threshold]

def filter_dictionary_by_percentage(dictionary, end_prcntg, strt_prcntg = 0):
    '''
    Filters a dictionary by a given percentage range.
    '''
    total_items = len(dictionary)
    
    start_index = int(total_items * (strt_prcntg / 100))
    end_index = int(total_items * (end_prcntg / 100))
    
    items = list(dictionary.items())
    
    return dict(items[start_index:end_index])