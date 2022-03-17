import brisk
from brisk.data_importer import raw

def import_imu():

    print('Import data to folder:')
    print(f'\t{brisk.out_dir}')
    raw.import_imu_data()

    return