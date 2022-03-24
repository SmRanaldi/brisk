import brisk
from brisk.data_importer import imu

def import_imu():

    print('Import data to folder:')
    print(f'\t{brisk.out_dir}')
    imu.import_imu_data()

    return